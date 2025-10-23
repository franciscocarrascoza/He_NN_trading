#include "io/TrajectoryReader.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "util/Logging.hpp"

namespace bls {

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string extensionOf(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return "";
    }
    return toLower(path.substr(pos + 1));
}

class UnavailableReader : public TrajectoryReader {
  public:
    explicit UnavailableReader(std::string message) : message_(std::move(message)) {}

    bool open(const std::string& path) override {
        logError(message_ + ": " + path);
        return false;
    }

    bool read(Frame&) override { return false; }

    void close() override {}

  private:
    std::string message_;
};

}  // namespace

std::unique_ptr<TrajectoryReader> createGroReader();
std::unique_ptr<TrajectoryReader> createPdbReader();
std::unique_ptr<TrajectoryReader> createXtcReader();
std::unique_ptr<TrajectoryReader> createTrrReader();
std::unique_ptr<TrajectoryReader> createTngReader();
Topology loadGroTopology(const std::string& path);
Topology loadPdbTopology(const std::string& path);

std::unique_ptr<TrajectoryReader> createTrajectoryReader(const std::string& path, const std::string& overrideFormat) {
    std::string fmt = overrideFormat.empty() ? extensionOf(path) : toLower(overrideFormat);
    if (fmt == "gro") {
        return createGroReader();
    }
    if (fmt == "pdb") {
        return createPdbReader();
    }
    if (fmt == "xtc") {
#ifdef USE_XDRFILE
        return createXtcReader();
#else
        return std::make_unique<UnavailableReader>("XTC support disabled at build time");
#endif
    }
    if (fmt == "trr") {
#ifdef USE_XDRFILE
        return createTrrReader();
#else
        return std::make_unique<UnavailableReader>("TRR support disabled at build time");
#endif
    }
    if (fmt == "tng") {
#ifdef USE_TNG
        return createTngReader();
#else
        return std::make_unique<UnavailableReader>("TNG support disabled at build time");
#endif
    }
    logError("Unsupported trajectory format: " + fmt);
    return nullptr;
}

Topology loadTopology(const std::string& path) {
    std::string ext = extensionOf(path);
    if (ext == "gro") {
        return loadGroTopology(path);
    }
    if (ext == "pdb") {
        return loadPdbTopology(path);
    }
    Topology topo;
    logWarn("No topology loader for format: " + ext);
    return topo;
}

}  // namespace bls

