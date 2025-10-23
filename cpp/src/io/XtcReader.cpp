#include "io/TrajectoryReader.hpp"

#include <string>
#include <vector>

#include "util/Logging.hpp"

namespace bls {

#ifdef USE_XDRFILE
extern "C" {
#include <xdrfile.h>
#include <xdrfile_xtc.h>
}

class XtcReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        path_ = path;
        if (read_xtc_natoms(path.c_str(), &natoms_) != exdrOK) {
            logError("Failed to read number of atoms from XTC: " + path);
            return false;
        }
        file_ = xdrfile_open(path.c_str(), "r");
        if (!file_) {
            logError("Failed to open XTC file: " + path);
            return false;
        }
        buffer_.resize(static_cast<std::size_t>(natoms_) * 3);
        return true;
    }

    bool read(Frame& frame) override {
        if (!file_) {
            return false;
        }
        matrix box;
        int step = 0;
        float time = 0.0f;
        float prec = 0.0f;
        auto coords = reinterpret_cast<rvec*>(buffer_.data());
        int status = read_xtc(file_, natoms_, &step, &time, box, coords, &prec);
        if (status != exdrOK) {
            return false;
        }
        frame.natoms = natoms_;
        frame.xyz.resize(natoms_);
        for (int i = 0; i < natoms_; ++i) {
            frame.xyz[i] = Vec3{coords[i][0], coords[i][1], coords[i][2]};
        }
        frame.box = Mat3{Vec3{box[0][0], box[0][1], box[0][2]}, Vec3{box[1][0], box[1][1], box[1][2]}, Vec3{box[2][0], box[2][1], box[2][2]}};
        frame.time = static_cast<double>(time);
        return true;
    }

    void close() override {
        if (file_) {
            xdrfile_close(file_);
            file_ = nullptr;
        }
        buffer_.clear();
    }

  private:
    std::string path_;
    XDRFILE* file_{nullptr};
    int natoms_{0};
    std::vector<float> buffer_;
};

std::unique_ptr<TrajectoryReader> createXtcReader() {
    return std::make_unique<XtcReader>();
}

#else

class XtcUnavailableReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        logError("BLS analyzer was built without XTC support: " + path);
        return false;
    }
    bool read(Frame&) override { return false; }
    void close() override {}
};

std::unique_ptr<TrajectoryReader> createXtcReader() {
    return std::make_unique<XtcUnavailableReader>();
}

#endif

}  // namespace bls

