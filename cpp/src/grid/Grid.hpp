#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bls/Parameters.hpp"
#include "io/TrajectoryReader.hpp"

namespace bls {

struct GridDimensions {
    int nx{0};
    int ny{0};
    int nz{0};
    Vec3 spacing{1.0, 1.0, 1.0};
    Mat3 box;
    Mat3 inverseBox;
    Vec3 origin{0.0, 0.0, 0.0};
};

class Grid {
  public:
    void initialize(const Mat3& box, double targetSpacing, const Vec3& origin = Vec3{0.0, 0.0, 0.0});
    void clear();

    int nx() const { return dims_.nx; }
    int ny() const { return dims_.ny; }
    int nz() const { return dims_.nz; }
    std::size_t size() const { return occupancy_.size(); }

    const GridDimensions& dims() const { return dims_; }
    const std::vector<uint8_t>& occupancy() const { return occupancy_; }
    std::vector<uint8_t>& occupancy() { return occupancy_; }
    std::vector<uint8_t>& visited() { return visited_; }
    const std::vector<uint8_t>& visited() const { return visited_; }

    void resetVisited();
    int index(int x, int y, int z) const;

  private:
    GridDimensions dims_;
    std::vector<uint8_t> occupancy_;
    std::vector<uint8_t> visited_;
};

void voxelize(const Frame& frame,
              const std::vector<int>& selection,
              const std::string& nameFilter,
              const Topology& topology,
              const BLSParameters& params,
              Grid& grid);

}  // namespace bls

