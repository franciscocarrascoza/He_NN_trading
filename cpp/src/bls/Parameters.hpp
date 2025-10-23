#pragma once

#include <string>
#include <vector>

#include "util/Math.hpp"

namespace bls {

enum class LatticeType { Cubic, Hexagonal, Triclinic };
enum class CenteringType { P, F, I };
enum class OccupancyRule { Any, All };

enum class ConnectivityMode { Six = 6, Eighteen = 18, TwentySix = 26 };

struct BoxSpec {
    bool autoBox{true};
    Vec3 lower{0.0, 0.0, 0.0};
    Vec3 upper{0.0, 0.0, 0.0};
};

struct GroupSpec {
    bool all{true};
    std::vector<int> indices;
    std::string nameFilter;
};

struct BLSParameters {
    GroupSpec group;
    BoxSpec box;
    double gridSpacing{0.25};
    ConnectivityMode connectivity{ConnectivityMode::Six};
    double alpha{0.7};
    int skip{3};
    double dnn{0.0};
    std::vector<double> radii;
    bool radiiAreLengths{true};
    LatticeType lattice{LatticeType::Cubic};
    CenteringType centering{CenteringType::F};
    double hex_c_over_a{1.633};
    Vec3 triclinic_abc{1.0, 1.2, 1.4};
    Vec3 triclinic_angles{90.0, 90.0, 90.0};
    int stride{1};
    OccupancyRule occupancy{OccupancyRule::Any};
    double cutoff{0.0};
    std::vector<std::string> outputs;
};

}  // namespace bls

