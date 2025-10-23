#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <unistd.h>

namespace bls {

inline std::size_t currentRSS() {
    std::ifstream statm("/proc/self/statm");
    if (!statm) {
        return 0;
    }
    std::size_t total = 0;
    statm >> total;
    std::size_t resident = 0;
    statm >> resident;
    long page_size = sysconf(_SC_PAGESIZE);
    return resident * static_cast<std::size_t>(page_size);
}

}  // namespace bls

