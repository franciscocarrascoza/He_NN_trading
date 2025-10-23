#include "bls/BLS.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "grid/Grid.hpp"
#include "lattice/Basis.hpp"
#include "refine/SkipDFS.hpp"
#include "util/Logging.hpp"

namespace bls {

namespace {

double computeDnnVoxel(const BLSParameters& params) {
    if (params.dnn > 0.0) {
        return params.dnn;
    }
    if (!params.radii.empty()) {
        double minPair = std::numeric_limits<double>::max();
        for (double r1 : params.radii) {
            for (double r2 : params.radii) {
                minPair = std::min(minPair, r1 + r2);
            }
        }
        double dnnLength = params.alpha * minPair;
        return std::max(1.0, dnnLength / params.gridSpacing);
    }
    return std::max(1.0, std::ceil(1.0 / params.gridSpacing));
}

std::string latticeToString(LatticeType type) {
    switch (type) {
        case LatticeType::Cubic:
            return "cubic";
        case LatticeType::Hexagonal:
            return "hexagonal";
        case LatticeType::Triclinic:
            return "triclinic";
    }
    return "cubic";
}

std::string centeringToString(CenteringType centering) {
    switch (centering) {
        case CenteringType::P:
            return "P";
        case CenteringType::F:
            return "F";
        case CenteringType::I:
            return "I";
    }
    return "P";
}

}  // namespace

BLSAnalyzer::BLSAnalyzer(const BLSParameters& params) : params_(params) {
    baseSelection_ = params_.group.indices;
    baseSelection_.erase(std::remove_if(baseSelection_.begin(), baseSelection_.end(), [](int idx) { return idx < 0; }), baseSelection_.end());
    std::sort(baseSelection_.begin(), baseSelection_.end());
    baseSelection_.erase(std::unique(baseSelection_.begin(), baseSelection_.end()), baseSelection_.end());
    directions_ = connectivityDirections(params_.connectivity);
    centeringOffsets_ = centeringOffsets(params_.centering);
    recordClusterSizes_ = std::find(params_.outputs.begin(), params_.outputs.end(), "CLUSTER_SIZES") != params_.outputs.end();
    Mat3 unitBasis = unitLatticeBasis(params_);
    double dmin = shortestUnitDistance(unitBasis, centeringOffsets_);
    dnnVoxel_ = computeDnnVoxel(params_);
    if (dmin < 1e-6) {
        throw std::runtime_error("Degenerate lattice basis; nearest neighbour distance is zero");
    }
    double scale = dnnVoxel_ / dmin;
    scaledBasis_ = unitBasis * scale;
}

Mat3 BLSAnalyzer::buildBox(const Frame& frame) const {
    if (params_.box.autoBox) {
        double det = std::abs(frame.box.determinant());
        if (det < 1e-6) {
            double cube = std::cbrt(static_cast<double>(std::max(frame.natoms, 1))) * params_.gridSpacing;
            return Mat3{Vec3{cube, 0.0, 0.0}, Vec3{0.0, cube, 0.0}, Vec3{0.0, 0.0, cube}};
        }
        return frame.box;
    }
    Vec3 lengths{params_.box.upper.x - params_.box.lower.x,
                 params_.box.upper.y - params_.box.lower.y,
                 params_.box.upper.z - params_.box.lower.z};
    if (lengths.x <= 0.0) lengths.x = params_.gridSpacing * 10.0;
    if (lengths.y <= 0.0) lengths.y = params_.gridSpacing * 10.0;
    if (lengths.z <= 0.0) lengths.z = params_.gridSpacing * 10.0;
    return Mat3{Vec3{lengths.x, 0.0, 0.0}, Vec3{0.0, lengths.y, 0.0}, Vec3{0.0, 0.0, lengths.z}};
}

std::vector<int> BLSAnalyzer::prepareSelection(const Frame& frame) const {
    std::vector<int> indices = baseSelection_;
    indices.erase(std::remove_if(indices.begin(), indices.end(), [&](int idx) { return idx < 0 || idx >= frame.natoms; }), indices.end());
    return indices;
}

FrameMetrics BLSAnalyzer::analyzeFrame(int frameIndex, const Frame& frame, const Topology& topology) {
    FrameMetrics metrics;
    metrics.frameIndex = frameIndex;
    metrics.time_ps = frame.time;
    metrics.natoms = frame.natoms;
    metrics.lattice = latticeToString(params_.lattice);
    metrics.centering = centeringToString(params_.centering);
    Timer timer;
    timer.start();

    Mat3 box = buildBox(frame);
    Vec3 origin = params_.box.autoBox ? Vec3{0.0, 0.0, 0.0} : params_.box.lower;
    grid_.initialize(box, params_.gridSpacing, origin);
    auto selection = prepareSelection(frame);
    voxelize(frame, selection, params_.group.nameFilter, topology, params_, grid_);

    metrics.nx = grid_.nx();
    metrics.ny = grid_.ny();
    metrics.nz = grid_.nz();
    metrics.dnn_vox = dnnVoxel_;

    SeedEnumerationResult seeds = enumerateSeeds(grid_.nx(), grid_.ny(), grid_.nz(), scaledBasis_, centeringOffsets_);
    metrics.seeds = seeds.totalSites;

    grid_.resetVisited();
    auto& occupancy = grid_.occupancy();
    for (int seed : seeds.seeds) {
        if (seed < 0 || seed >= static_cast<int>(grid_.size())) {
            continue;
        }
        if (occupancy[seed]) {
            ++metrics.seedHits;
            int cluster = skipDFS(grid_, seed, std::max(1, params_.skip), directions_);
            if (cluster > 0) {
                ++metrics.nclusters;
                metrics.maxCluster = std::max(metrics.maxCluster, cluster);
                metrics.refinedVoxels += cluster;
                if (recordClusterSizes_) {
                    metrics.clusterSizes.push_back(cluster);
                }
            }
        }
    }

    metrics.elapsed_ms = timer.elapsedMilliseconds();
    return metrics;
}

}  // namespace bls

