#include <cmath>
#include <iostream>

#include "bls/BLS.hpp"
#include "lattice/Basis.hpp"
#include "refine/SkipDFS.hpp"

using namespace bls;

bool approxEqual(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

int main() {
    bool ok = true;

    // Geometry test: FCC nearest neighbour distance.
    BLSParameters geomParams;
    geomParams.lattice = LatticeType::Cubic;
    geomParams.centering = CenteringType::F;
    Mat3 geomBasis = unitLatticeBasis(geomParams);
    auto geomOffsets = centeringOffsets(geomParams.centering);
    double dmin = shortestUnitDistance(geomBasis, geomOffsets);
    if (!approxEqual(dmin, std::sqrt(0.5))) {
        std::cerr << "Geometry test failed: expected sqrt(0.5), got " << dmin << std::endl;
        ok = false;
    }

    // Connectivity test.
    Grid grid;
    Mat3 box{Vec3{3.0, 0.0, 0.0}, Vec3{0.0, 3.0, 0.0}, Vec3{0.0, 0.0, 3.0}};
    grid.initialize(box, 1.0);
    grid.clear();
    auto index = [&](int x, int y, int z) { return grid.index(x, y, z); };
    grid.occupancy()[index(0, 0, 0)] = 1;
    grid.occupancy()[index(1, 1, 0)] = 1;
    auto dirs6 = connectivityDirections(ConnectivityMode::Six);
    int cluster6 = skipDFS(grid, index(0, 0, 0), 1, dirs6);
    if (cluster6 != 1) {
        std::cerr << "Connectivity test (6) failed: expected 1 got " << cluster6 << std::endl;
        ok = false;
    }
    grid.resetVisited();
    auto dirs26 = connectivityDirections(ConnectivityMode::TwentySix);
    int cluster26 = skipDFS(grid, index(0, 0, 0), 1, dirs26);
    if (cluster26 != 2) {
        std::cerr << "Connectivity test (26) failed: expected 2 got " << cluster26 << std::endl;
        ok = false;
    }

    // PBC test.
    Grid pbcGrid;
    Mat3 pbcBox{Vec3{2.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
    pbcGrid.initialize(pbcBox, 1.0);
    pbcGrid.clear();
    pbcGrid.occupancy()[pbcGrid.index(0, 0, 0)] = 1;
    pbcGrid.occupancy()[pbcGrid.index(1, 0, 0)] = 1;
    int clusterPbc = skipDFS(pbcGrid, pbcGrid.index(0, 0, 0), 2, dirs6);
    if (clusterPbc != 2) {
        std::cerr << "PBC test failed: expected 2 got " << clusterPbc << std::endl;
        ok = false;
    }

    // Determinism test.
    BLSParameters params;
    params.gridSpacing = 1.0;
    params.skip = 2;
    params.lattice = LatticeType::Cubic;
    params.centering = CenteringType::P;
    params.connectivity = ConnectivityMode::Six;
    params.alpha = 0.7;
    params.dnn = 1.0;
    BLSAnalyzer analyzer(params);
    Frame frame;
    frame.time = 0.0;
    frame.natoms = 2;
    frame.xyz = {Vec3{0.25, 0.25, 0.25}, Vec3{1.25, 0.25, 0.25}};
    frame.box = Mat3{Vec3{2.0, 0.0, 0.0}, Vec3{0.0, 2.0, 0.0}, Vec3{0.0, 0.0, 2.0}};
    Topology topo;
    FrameMetrics m1 = analyzer.analyzeFrame(0, frame, topo);
    analyzer.analyzeFrame(1, frame, topo);
    FrameMetrics m2 = analyzer.analyzeFrame(2, frame, topo);
    if (m1.nclusters != m2.nclusters || m1.maxCluster != m2.maxCluster) {
        std::cerr << "Determinism test failed." << std::endl;
        ok = false;
    }

    return ok ? 0 : 1;
}

