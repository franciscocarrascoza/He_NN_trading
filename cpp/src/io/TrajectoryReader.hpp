#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "util/Math.hpp"

namespace bls {

struct Frame {
    std::vector<Vec3> xyz;
    Mat3 box;
    int natoms{0};
    double time{0.0};
};

class TrajectoryReader {
  public:
    virtual ~TrajectoryReader() = default;
    virtual bool open(const std::string& path) = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void close() = 0;
};

std::unique_ptr<TrajectoryReader> createTrajectoryReader(const std::string& path, const std::string& overrideFormat = "");

struct TopologyAtom {
    std::string name;
    int index{0};
};

struct Topology {
    std::vector<TopologyAtom> atoms;
};

Topology loadTopology(const std::string& path);

}  // namespace bls

