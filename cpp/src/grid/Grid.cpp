#include "grid/Grid.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>

#include "util/Logging.hpp"

namespace bls {

void Grid::initialize(const Mat3& box, double targetSpacing, const Vec3& origin) {
    dims_.box = box;
    dims_.inverseBox = inverse(box);
    dims_.origin = origin;
    auto length = [](const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); };
    double spacing = targetSpacing > 0 ? targetSpacing : 0.25;
    dims_.nx = std::max(1, static_cast<int>(std::ceil(length(box.cols[0]) / spacing)));
    dims_.ny = std::max(1, static_cast<int>(std::ceil(length(box.cols[1]) / spacing)));
    dims_.nz = std::max(1, static_cast<int>(std::ceil(length(box.cols[2]) / spacing)));
    dims_.spacing = Vec3{length(box.cols[0]) / dims_.nx, length(box.cols[1]) / dims_.ny, length(box.cols[2]) / dims_.nz};
    std::size_t total = static_cast<std::size_t>(dims_.nx) * static_cast<std::size_t>(dims_.ny) * static_cast<std::size_t>(dims_.nz);
    occupancy_.assign(total, 0);
    visited_.assign(total, 0);
}

void Grid::clear() {
    std::fill(occupancy_.begin(), occupancy_.end(), 0);
    std::fill(visited_.begin(), visited_.end(), 0);
}

void Grid::resetVisited() {
    std::fill(visited_.begin(), visited_.end(), 0);
}

int Grid::index(int x, int y, int z) const {
    return x + dims_.nx * (y + dims_.ny * z);
}

namespace {

int wrapIndex(int value, int max) {
    if (max <= 0) return 0;
    int mod = value % max;
    if (mod < 0) mod += max;
    return mod;
}

Vec3 wrapPosition(const Vec3& pos, const GridDimensions& dims) {
    Vec3 fractional = dims.inverseBox * (pos - dims.origin);
    fractional.x -= std::floor(fractional.x);
    fractional.y -= std::floor(fractional.y);
    fractional.z -= std::floor(fractional.z);
    return dims.box * fractional;
}

int locate(const Vec3& pos, const GridDimensions& dims, int axis) {
    Vec3 fractional = dims.inverseBox * (pos - dims.origin);
    fractional.x -= std::floor(fractional.x);
    fractional.y -= std::floor(fractional.y);
    fractional.z -= std::floor(fractional.z);
    switch (axis) {
        case 0: {
            int idx = static_cast<int>(std::floor(fractional.x * dims.nx));
            if (idx == dims.nx) idx = 0;
            return idx;
        }
        case 1: {
            int idx = static_cast<int>(std::floor(fractional.y * dims.ny));
            if (idx == dims.ny) idx = 0;
            return idx;
        }
        default: {
            int idx = static_cast<int>(std::floor(fractional.z * dims.nz));
            if (idx == dims.nz) idx = 0;
            return idx;
        }
    }
}

double minSpacing(const GridDimensions& dims) {
    return std::min({dims.spacing.x, dims.spacing.y, dims.spacing.z});
}

}  // namespace

void voxelize(const Frame& frame,
              const std::vector<int>& selection,
              const std::string& nameFilter,
              const Topology& topology,
              const BLSParameters& params,
              Grid& grid) {
    grid.clear();
    std::vector<int> atoms;
    if (!selection.empty()) {
        atoms = selection;
    } else {
        atoms.resize(frame.natoms);
        std::iota(atoms.begin(), atoms.end(), 0);
    }
    if (!nameFilter.empty() && !topology.atoms.empty()) {
        std::vector<int> filtered;
        std::string upperFilter = nameFilter;
        std::transform(upperFilter.begin(), upperFilter.end(), upperFilter.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        for (int idx : atoms) {
            if (idx < 0 || idx >= static_cast<int>(topology.atoms.size())) continue;
            std::string atomName = topology.atoms[idx].name;
            std::transform(atomName.begin(), atomName.end(), atomName.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (atomName == upperFilter) {
                filtered.push_back(idx);
            }
        }
        if (!filtered.empty()) {
            atoms.swap(filtered);
        }
    }

    const auto& dims = grid.dims();
    double cutoff = params.cutoff;
    double minSpace = minSpacing(dims);
    int extraX = static_cast<int>(std::ceil(cutoff / std::max(dims.spacing.x, 1e-6)));
    int extraY = static_cast<int>(std::ceil(cutoff / std::max(dims.spacing.y, 1e-6)));
    int extraZ = static_cast<int>(std::ceil(cutoff / std::max(dims.spacing.z, 1e-6)));

#pragma omp parallel for schedule(static) if (atoms.size() > 256 && extraX + extraY + extraZ > 0)
    for (std::size_t n = 0; n < atoms.size(); ++n) {
        int atomIndex = atoms[n];
        if (atomIndex < 0 || atomIndex >= frame.natoms) {
            continue;
        }
        Vec3 pos = wrapPosition(frame.xyz[atomIndex], dims);
        int ix = locate(pos, dims, 0);
        int iy = locate(pos, dims, 1);
        int iz = locate(pos, dims, 2);
        auto mark = [&](int x, int y, int z) {
            x = wrapIndex(x, dims.nx);
            y = wrapIndex(y, dims.ny);
            z = wrapIndex(z, dims.nz);
            int idx = grid.index(x, y, z);
#pragma omp atomic write
            grid.occupancy()[static_cast<std::size_t>(idx)] = 1;
        };

        if (params.occupancy == OccupancyRule::All) {
            double radius = cutoff;
            if (radius < 0.5 * minSpace) {
                double fx = (dims.inverseBox * pos).x * dims.nx - ix;
                double fy = (dims.inverseBox * pos).y * dims.ny - iy;
                double fz = (dims.inverseBox * pos).z * dims.nz - iz;
                double dx = std::min(fx * dims.spacing.x, (1.0 - fx) * dims.spacing.x);
                double dy = std::min(fy * dims.spacing.y, (1.0 - fy) * dims.spacing.y);
                double dz = std::min(fz * dims.spacing.z, (1.0 - fz) * dims.spacing.z);
                if (dx >= radius && dy >= radius && dz >= radius) {
                    mark(ix, iy, iz);
                }
            }
        } else {
            for (int dx = -extraX; dx <= extraX; ++dx) {
                for (int dy = -extraY; dy <= extraY; ++dy) {
                    for (int dz = -extraZ; dz <= extraZ; ++dz) {
                        double dist = std::sqrt((dx * dims.spacing.x) * (dx * dims.spacing.x) +
                                                (dy * dims.spacing.y) * (dy * dims.spacing.y) +
                                                (dz * dims.spacing.z) * (dz * dims.spacing.z));
                        if (dist <= cutoff + 1e-6) {
                            mark(ix + dx, iy + dy, iz + dz);
                        }
                    }
                }
            }
            mark(ix, iy, iz);
        }
    }
}

}  // namespace bls

