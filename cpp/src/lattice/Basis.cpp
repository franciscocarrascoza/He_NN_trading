#include "lattice/Basis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace bls {

namespace {

Mat3 cubicBasis() {
    return Mat3{Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
}

Mat3 hexagonalBasis(double c_over_a) {
    const double sqrt3_over2 = std::sqrt(3.0) / 2.0;
    return Mat3{Vec3{1.0, 0.0, 0.0}, Vec3{-0.5, sqrt3_over2, 0.0}, Vec3{0.0, 0.0, c_over_a}};
}

Mat3 triclinicBasis(const Vec3& abc, const Vec3& angles) {
    const double deg2rad = 3.14159265358979323846 / 180.0;
    double alpha = angles.x * deg2rad;
    double beta = angles.y * deg2rad;
    double gamma = angles.z * deg2rad;
    double ca = std::cos(alpha);
    double cb = std::cos(beta);
    double cg = std::cos(gamma);
    double sg = std::sin(gamma);
    Vec3 a_vec{abc.x, 0.0, 0.0};
    Vec3 b_vec{abc.y * cg, abc.y * sg, 0.0};
    double cx = abc.z * cb;
    double cy = abc.z * (ca - cb * cg) / sg;
    double cz = std::sqrt(std::max(0.0, abc.z * abc.z - cx * cx - cy * cy));
    Vec3 c_vec{cx, cy, cz};
    return Mat3{a_vec, b_vec, c_vec};
}

}  // namespace

Mat3 unitLatticeBasis(const BLSParameters& params) {
    switch (params.lattice) {
        case LatticeType::Cubic:
            return cubicBasis();
        case LatticeType::Hexagonal:
            return hexagonalBasis(params.hex_c_over_a);
        case LatticeType::Triclinic:
            return triclinicBasis(params.triclinic_abc, params.triclinic_angles);
    }
    return cubicBasis();
}

std::vector<Vec3> centeringOffsets(CenteringType centering) {
    switch (centering) {
        case CenteringType::P:
            return {Vec3{0.0, 0.0, 0.0}};
        case CenteringType::F:
            return {Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 0.5, 0.5}, Vec3{0.5, 0.0, 0.5}, Vec3{0.5, 0.5, 0.0}};
        case CenteringType::I:
            return {Vec3{0.0, 0.0, 0.0}, Vec3{0.5, 0.5, 0.5}};
    }
    return {Vec3{0.0, 0.0, 0.0}};
}

double shortestUnitDistance(const Mat3& basis, const std::vector<Vec3>& offsets) {
    double minDist = std::numeric_limits<double>::max();
    for (const auto& offsetA : offsets) {
        for (const auto& offsetB : offsets) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        Vec3 latticeVec{static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
                        Vec3 delta = basis * (latticeVec + offsetA - offsetB);
                        double dist = delta.norm();
                        if (dist > 1e-8) {
                            minDist = std::min(minDist, dist);
                        }
                    }
                }
            }
        }
    }
    return minDist;
}

SeedEnumerationResult enumerateSeeds(int nx, int ny, int nz, const Mat3& basis, const std::vector<Vec3>& offsets) {
    SeedEnumerationResult result;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return result;
    }
    std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    std::vector<uint8_t> mask(total, 0);

    auto column = [&](int idx) { return basis.cols[static_cast<std::size_t>(idx)]; };
    auto stepMagnitude = [&](int idx) {
        Vec3 col = column(idx);
        return std::max({std::fabs(col.x), std::fabs(col.y), std::fabs(col.z), 1e-6});
    };

    int rangeI = static_cast<int>(std::ceil((nx + 2.0) / stepMagnitude(0))) + 1;
    int rangeJ = static_cast<int>(std::ceil((ny + 2.0) / stepMagnitude(1))) + 1;
    int rangeK = static_cast<int>(std::ceil((nz + 2.0) / stepMagnitude(2))) + 1;

    for (int i = -rangeI; i <= rangeI; ++i) {
        for (int j = -rangeJ; j <= rangeJ; ++j) {
            for (int k = -rangeK; k <= rangeK; ++k) {
                Vec3 latticeIndex{static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
                for (const auto& offset : offsets) {
                    Vec3 site = basis * (latticeIndex + offset);
                    long px = std::lround(site.x);
                    long py = std::lround(site.y);
                    long pz = std::lround(site.z);
                    if (px < 0 || py < 0 || pz < 0) {
                        continue;
                    }
                    if (px >= nx || py >= ny || pz >= nz) {
                        continue;
                    }
                    std::size_t index = static_cast<std::size_t>(px) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(py) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(pz));
                    if (!mask[index]) {
                        mask[index] = 1;
                        result.seeds.push_back(static_cast<int>(index));
                    }
                }
            }
        }
    }
    result.totalSites = result.seeds.size();
    return result;
}

}  // namespace bls
