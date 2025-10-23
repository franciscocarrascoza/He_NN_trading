#include "io/TrajectoryReader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "util/Logging.hpp"

namespace bls {

namespace {

Mat3 boxFromCell(double a, double b, double c, double alpha, double beta, double gamma) {
    const double deg2rad = 3.14159265358979323846 / 180.0;
    alpha *= deg2rad;
    beta *= deg2rad;
    gamma *= deg2rad;
    double ca = std::cos(alpha);
    double cb = std::cos(beta);
    double cg = std::cos(gamma);
    double sg = std::sin(gamma);

    Vec3 a_vec{a, 0.0, 0.0};
    Vec3 b_vec{b * cg, b * sg, 0.0};
    double cx = c * cb;
    double cy = c * (ca - cb * cg) / sg;
    double cz = std::sqrt(std::max(0.0, c * c - cx * cx - cy * cy));
    Vec3 c_vec{cx, cy, cz};
    return Mat3{a_vec, b_vec, c_vec};
}

}  // namespace

class PdbReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        file_.open(path);
        if (!file_) {
            logError("Failed to open PDB file: " + path);
            return false;
        }
        loadAllFrames();
        current_ = 0;
        return !frames_.empty();
    }

    bool read(Frame& frame) override {
        if (current_ >= frames_.size()) {
            return false;
        }
        frame = frames_[current_++];
        return true;
    }

    void close() override {
        frames_.clear();
        current_ = 0;
        file_.close();
    }

  private:
    void loadAllFrames() {
        frames_.clear();
        std::string line;
        Frame frame;
        frame.time = 0.0;
        frame.natoms = 0;
        Mat3 currentBox{Vec3{10.0, 0.0, 0.0}, Vec3{0.0, 10.0, 0.0}, Vec3{0.0, 0.0, 10.0}};
        frame.box = currentBox;
        while (std::getline(file_, line)) {
            if (line.rfind("MODEL", 0) == 0) {
                if (frame.natoms > 0) {
                    frames_.push_back(frame);
                    frame = Frame{};
                    frame.box = currentBox;
                    frame.time = static_cast<double>(frames_.size());
                }
                continue;
            }
            if (line.rfind("ENDMDL", 0) == 0) {
                if (frame.natoms > 0) {
                    frames_.push_back(frame);
                    frame = Frame{};
                    frame.box = currentBox;
                }
                continue;
            }
            if (line.rfind("CRYST1", 0) == 0) {
                double a = std::atof(line.substr(6, 9).c_str());
                double b = std::atof(line.substr(15, 9).c_str());
                double c = std::atof(line.substr(24, 9).c_str());
                double alpha = std::atof(line.substr(33, 7).c_str());
                double beta = std::atof(line.substr(40, 7).c_str());
                double gamma = std::atof(line.substr(47, 7).c_str());
                currentBox = boxFromCell(a, b, c, alpha, beta, gamma);
                if (frame.box.determinant() == 0.0) {
                    frame.box = currentBox;
                }
                continue;
            }
            if (line.size() >= 54 && (line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0)) {
                double x = std::atof(line.substr(30, 8).c_str());
                double y = std::atof(line.substr(38, 8).c_str());
                double z = std::atof(line.substr(46, 8).c_str());
                frame.xyz.emplace_back(x, y, z);
                frame.natoms = static_cast<int>(frame.xyz.size());
                if (frame.box.determinant() == 0.0) {
                    frame.box = currentBox;
                }
            }
        }
        if (frame.natoms > 0) {
            if (frame.box.determinant() == 0.0) {
                frame.box = currentBox;
            }
            frames_.push_back(frame);
        }
    }

    std::ifstream file_;
    std::vector<Frame> frames_;
    std::size_t current_{0};
};

std::unique_ptr<TrajectoryReader> createPdbReader() {
    return std::make_unique<PdbReader>();
}

Topology loadPdbTopology(const std::string& path) {
    std::ifstream file(path);
    Topology topo;
    if (!file) {
        logWarn("Failed to open PDB topology: " + path);
        return topo;
    }
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        if (line.size() >= 54 && (line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0)) {
            std::string name = line.substr(12, 4);
            name.erase(0, name.find_first_not_of(' '));
            name.erase(name.find_last_not_of(' ') + 1);
            topo.atoms.push_back(TopologyAtom{name, index++});
        }
    }
    return topo;
}

}  // namespace bls

