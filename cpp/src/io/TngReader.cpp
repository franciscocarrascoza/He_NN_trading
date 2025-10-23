#include "io/TrajectoryReader.hpp"

#include <cstdint>
#include <string>

#include "util/Logging.hpp"

namespace bls {

#ifdef USE_TNG
extern "C" {
#include <tng/tng_io.h>
}

class TngReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        if (tng_trajectory_init(&traj_) != TNG_SUCCESS) {
            logError("Failed to initialise TNG trajectory");
            return false;
        }
        if (tng_trajectory_open(traj_, path.c_str(), 'r') != TNG_SUCCESS) {
            logError("Failed to open TNG file: " + path);
            return false;
        }
        if (tng_num_particles_get(traj_, &natoms_) != TNG_SUCCESS) {
            logError("Failed to query number of atoms in TNG file");
            return false;
        }
        return true;
    }

    bool read(Frame& frame) override {
        if (!traj_) {
            return false;
        }
        if (tng_frame_current_get(traj_, &currentFrame_) != TNG_SUCCESS) {
            return false;
        }
        if (tng_frame_set(traj_, currentFrame_) != TNG_SUCCESS) {
            return false;
        }
        double time = 0.0;
        if (tng_time_get(traj_, &time) != TNG_SUCCESS) {
            time = static_cast<double>(currentFrame_);
        }
        frame.time = time;
        frame.natoms = static_cast<int>(natoms_);
        frame.xyz.resize(frame.natoms);
        frame.box = Mat3{};
        float* coords = nullptr;
        int64_t coordsCount = 0;
        if (tng_util_particle_data_get(traj_, TNG_TRAJ_POSITIONS, &coords, &coordsCount) != TNG_SUCCESS) {
            logError("Failed to read coordinates from TNG frame");
            return false;
        }
        if (coordsCount < static_cast<int64_t>(frame.natoms) * 3) {
            logError("TNG coordinate block is undersized");
            return false;
        }
        for (int i = 0; i < frame.natoms; ++i) {
            frame.xyz[i] = Vec3{coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]};
        }
        tng_free(coords);
        double boxData[9];
        if (tng_box_shape_get(traj_, boxData) == TNG_SUCCESS) {
            frame.box = Mat3{Vec3{boxData[0], boxData[3], boxData[4]}, Vec3{boxData[6], boxData[1], boxData[5]}, Vec3{boxData[7], boxData[8], boxData[2]}};
        }
        ++currentFrame_;
        return true;
    }

    void close() override {
        if (traj_) {
            tng_trajectory_close(&traj_);
            traj_ = nullptr;
        }
    }

  private:
    tng_trajectory_t traj_{nullptr};
    int64_t natoms_{0};
    int64_t currentFrame_{0};
};

std::unique_ptr<TrajectoryReader> createTngReader() {
    return std::make_unique<TngReader>();
}

#else

class TngUnavailableReader : public TrajectoryReader {
  public:
    bool open(const std::string& path) override {
        logError("BLS analyzer was built without TNG support: " + path);
        return false;
    }
    bool read(Frame&) override { return false; }
    void close() override {}
};

std::unique_ptr<TrajectoryReader> createTngReader() {
    return std::make_unique<TngUnavailableReader>();
}

#endif

}  // namespace bls

