#pragma once

#include <cstddef>
#include <vector>

#include "bls/Parameters.hpp"

namespace bls {

Mat3 unitLatticeBasis(const BLSParameters& params);
std::vector<Vec3> centeringOffsets(CenteringType centering);
double shortestUnitDistance(const Mat3& basis, const std::vector<Vec3>& offsets);

struct SeedEnumerationResult {
    std::vector<int> seeds;
    std::size_t totalSites{0};
};

SeedEnumerationResult enumerateSeeds(int nx, int ny, int nz, const Mat3& basis, const std::vector<Vec3>& offsets);

}  // namespace bls

