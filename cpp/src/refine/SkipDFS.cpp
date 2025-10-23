#include "refine/SkipDFS.hpp"

#include <stack>

namespace bls {

namespace {

int wrap(int value, int bound) {
    if (bound <= 0) return 0;
    int mod = value % bound;
    if (mod < 0) mod += bound;
    return mod;
}

std::array<int, 3> decode(int index, const Grid& grid) {
    int plane = grid.nx() * grid.ny();
    int z = index / plane;
    int remainder = index % plane;
    int y = remainder / grid.nx();
    int x = remainder % grid.nx();
    return {x, y, z};
}

}  // namespace

std::vector<std::array<int, 3>> connectivityDirections(ConnectivityMode mode) {
    std::vector<std::array<int, 3>> dirs;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                int manhattan = std::abs(dx) + std::abs(dy) + std::abs(dz);
                if (mode == ConnectivityMode::Six && manhattan == 1) {
                    dirs.push_back({dx, dy, dz});
                } else if (mode == ConnectivityMode::Eighteen && manhattan <= 2) {
                    dirs.push_back({dx, dy, dz});
                } else if (mode == ConnectivityMode::TwentySix) {
                    dirs.push_back({dx, dy, dz});
                }
            }
        }
    }
    return dirs;
}

int skipDFS(Grid& grid, int seedIndex, int skip, const std::vector<std::array<int, 3>>& directions) {
    if (seedIndex < 0 || seedIndex >= static_cast<int>(grid.size())) {
        return 0;
    }
    auto& occupancy = grid.occupancy();
    auto& visited = grid.visited();
    if (!occupancy[seedIndex] || visited[seedIndex]) {
        return 0;
    }
    std::stack<int> stack;
    stack.push(seedIndex);
    int clusterSize = 0;

    while (!stack.empty()) {
        int index = stack.top();
        stack.pop();
        if (visited[index]) {
            continue;
        }
        visited[index] = 1;
        ++clusterSize;
        auto coords = decode(index, grid);
        int x0 = coords[0];
        int y0 = coords[1];
        int z0 = coords[2];
        for (const auto& dir : directions) {
            int cx = x0;
            int cy = y0;
            int cz = z0;
            for (int step = 0; step < skip; ++step) {
                int nx = wrap(cx + dir[0], grid.nx());
                int ny = wrap(cy + dir[1], grid.ny());
                int nz = wrap(cz + dir[2], grid.nz());
                if (nx == cx && ny == cy && nz == cz) {
                    break;
                }
                int neighborIndex = grid.index(nx, ny, nz);
                if (!occupancy[neighborIndex]) {
                    break;
                }
                if (!visited[neighborIndex]) {
                    stack.push(neighborIndex);
                }
                cx = nx;
                cy = ny;
                cz = nz;
            }
        }
    }
    return clusterSize;
}

}  // namespace bls

