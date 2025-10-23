#pragma once

#include <array>
#include <vector>

#include "bls/Parameters.hpp"
#include "grid/Grid.hpp"

namespace bls {

std::vector<std::array<int, 3>> connectivityDirections(ConnectivityMode mode);
int skipDFS(Grid& grid, int seedIndex, int skip, const std::vector<std::array<int, 3>>& directions);

}  // namespace bls

