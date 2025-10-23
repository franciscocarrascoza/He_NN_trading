#include "io/TrajectoryReader.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "util/Logging.hpp"

namespace bls {

namespace {

Mat3 parseBox(const std::string& line) {
    std::istringstream iss(line);
    double values[9] = {0.0};
    int idx = 0;
    while (iss && idx < 9) {
        iss >> values[idx++];
    }
    if (idx >= 3) {
        // GRO boxes are in nm and can provide either 3 or 9 values.
        if (idx < 9) {
            return Mat3{Vec3{values[0], 0.0, 0.0}, Vec3{0.0, values[1], 0.0}, Vec3{0.0, 0.0, values[2]}};
        }
        // triclinic case
        return Mat3{
            Vec3{values[0], values[3], values[4]},
            Vec3{0.0, values[1], values[5]},
            Vec3{0.0, 0.0, values[2]}};
    }
    double fallback = static_cast<double>(line.size());
    return Mat3{Vec3{fallback, 0.0, 0.0}, Vec3{0.0, fallback, 0.0}, Vec3{0.0, 0.0, fallback}};
}

Vec3 parsePosition(const std::string& line) {
    // GRO uses fixed width fields; coordinates at 20-44 (nm).
    auto readField = [&](std::size_t start) {
        std::size_t end = std::min(line.size(), start + 8);
        std::string token = line.substr(start, end - start);
        return std::atof(token.c_str());
    };
    return Vec3{readField(20), readField(28), readField(36)};
}

}  // namespace

class GroReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        file_.open(path);
        if (!file_) {
            logError("Failed to open GRO file: " + path);
            return false;
        }
        frameIndex_ = 0;
        return true;
    }

    bool read(Frame& frame) override {
        if (!file_) {
            return false;
        }
        std::string title;
        if (!std::getline(file_, title)) {
            return false;
        }
        std::string natomsLine;
        if (!std::getline(file_, natomsLine)) {
            return false;
        }
        int natoms = std::stoi(natomsLine);
        frame.xyz.clear();
        frame.xyz.reserve(natoms);
        frame.natoms = natoms;

        std::string atomLine;
        for (int i = 0; i < natoms; ++i) {
            if (!std::getline(file_, atomLine)) {
                logError("Unexpected EOF in GRO atom block");
                return false;
            }
            frame.xyz.emplace_back(parsePosition(atomLine));
        }
        std::string boxLine;
        if (!std::getline(file_, boxLine)) {
            logError("Missing box line in GRO file");
            return false;
        }
        frame.box = parseBox(boxLine);
        frame.time = parseTime(title, frameIndex_);
        ++frameIndex_;
        return true;
    }

    void close() override {
        file_.close();
    }

  private:
    static double parseTime(const std::string& title, int index) {
        std::size_t pos = title.find("t=");
        if (pos != std::string::npos) {
            double time = std::atof(title.c_str() + pos + 2);
            return time;
        }
        return static_cast<double>(index);
    }

    std::ifstream file_;
    int frameIndex_{0};
};

std::unique_ptr<TrajectoryReader> createGroReader() {
    return std::make_unique<GroReader>();
}

Topology loadGroTopology(const std::string& path) {
    std::ifstream file(path);
    Topology topo;
    if (!file) {
        logWarn("Failed to open GRO topology: " + path);
        return topo;
    }
    std::string line;
    std::getline(file, line);  // title
    std::getline(file, line);  // natoms
    int natoms = line.empty() ? 0 : std::stoi(line);
    topo.atoms.reserve(natoms);
    for (int i = 0; i < natoms; ++i) {
        if (!std::getline(file, line)) {
            break;
        }
        if (line.size() < 15) {
            continue;
        }
        std::string name = line.substr(10, 5);
        name.erase(0, name.find_first_not_of(' '));
        name.erase(name.find_last_not_of(' ') + 1);
        topo.atoms.push_back(TopologyAtom{name, i});
    }
    return topo;
}

}  // namespace bls

