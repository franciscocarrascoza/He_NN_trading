#pragma once

#include <chrono>

namespace bls {

class Timer {
  public:
    using clock = std::chrono::steady_clock;

    void start() { start_ = clock::now(); }

    double elapsedMilliseconds() const {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - start_).count();
    }

  private:
    clock::time_point start_ = clock::now();
};

}  // namespace bls

