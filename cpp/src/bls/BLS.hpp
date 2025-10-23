#pragma once

#include <array>
#include <string>
#include <vector>

#include "bls/Parameters.hpp"
#include "grid/Grid.hpp"
#include "io/TrajectoryReader.hpp"
#include "util/Timer.hpp"

namespace bls {

struct FrameMetrics {
    int frameIndex{0};
    double time_ps{0.0};
    int natoms{0};
    int nx{0};
    int ny{0};
    int nz{0};
    double dnn_vox{0.0};
    std::string lattice;
    std::string centering;
    std::size_t seeds{0};
    int seedHits{0};
    int nclusters{0};
    int maxCluster{0};
    int refinedVoxels{0};
    double elapsed_ms{0.0};
    std::vector<int> clusterSizes;
};

class BLSAnalyzer {
  public:
    explicit BLSAnalyzer(const BLSParameters& params);

    FrameMetrics analyzeFrame(int frameIndex, const Frame& frame, const Topology& topology);

  private:
    BLSParameters params_;
    Grid grid_;
    std::vector<int> baseSelection_;
    std::vector<std::array<int, 3>> directions_;
    Mat3 scaledBasis_;
    std::vector<Vec3> centeringOffsets_;
    double dnnVoxel_{1.0};
    bool recordClusterSizes_{false};

    Mat3 buildBox(const Frame& frame) const;
    std::vector<int> prepareSelection(const Frame& frame) const;
};

}  // namespace bls

