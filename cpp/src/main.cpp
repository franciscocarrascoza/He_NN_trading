#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bls/BLS.hpp"
#include "config/Parser.hpp"
#include "io/TrajectoryReader.hpp"
#include "util/Logging.hpp"
#include "util/RSS.hpp"

namespace bls {

struct CLIOptions {
    std::string traj;
    std::string top;
    std::string config;
    std::string outCsv;
    std::string outJson;
    std::string comparePlumed;
    std::string formatOverride;
    int stride{1};
    int start{0};
    int stop{std::numeric_limits<int>::max()};
    int threads{1};
};

void printUsage() {
    std::cout << "Usage: bls_analyze --traj traj.xtc --conf bls.in [options]\n"
              << "Options:\n"
              << "  --top FILE            Topology file for selections (gro/pdb)\n"
              << "  --out FILE            CSV output file\n"
              << "  --json FILE           JSON lines output file\n"
              << "  --stride N            Analyze every Nth frame (default 1)\n"
              << "  --start N             First frame index to include (default 0)\n"
              << "  --stop N              Last frame index to include (default inf)\n"
              << "  --threads N           Number of OpenMP threads\n"
              << "  --format FMT          Force trajectory format (xtc|trr|tng|gro|pdb)\n"
              << "  --compare-plumed FILE Reference PLUMED metrics for comparison\n"
              << std::endl;
}

std::optional<std::string> requireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        return std::nullopt;
    }
    ++index;
    return std::string(argv[index]);
}

bool parseCLI(int argc, char** argv, CLIOptions& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--traj") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.traj = *value;
        } else if (arg == "--top") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.top = *value;
        } else if (arg == "--conf") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.config = *value;
        } else if (arg == "--out") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.outCsv = *value;
        } else if (arg == "--json") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.outJson = *value;
        } else if (arg == "--stride") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.stride = std::max(1, std::stoi(*value));
        } else if (arg == "--start") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.start = std::max(0, std::stoi(*value));
        } else if (arg == "--stop") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            if (*value == "inf" || *value == "INF") {
                options.stop = std::numeric_limits<int>::max();
            } else {
                options.stop = std::stoi(*value);
            }
        } else if (arg == "--threads") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.threads = std::max(1, std::stoi(*value));
        } else if (arg == "--format") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.formatOverride = *value;
        } else if (arg == "--compare-plumed") {
            auto value = requireValue(argc, argv, i);
            if (!value) return false;
            options.comparePlumed = *value;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return false;
        }
    }
    if (options.traj.empty() || options.config.empty()) {
        printUsage();
        return false;
    }
    return true;
}

struct ReferenceRow {
    double time{0.0};
    double maxCluster{0.0};
    double nclusters{0.0};
    double elapsedMs{0.0};
};

std::vector<std::string> split(const std::string& line, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(line);
    while (std::getline(iss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> splitWhitespace(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<ReferenceRow> loadReference(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open reference file: " + path);
    }
    std::vector<ReferenceRow> rows;
    std::vector<std::string> columns;
    char delim = ',';
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto fieldsPos = line.find("FIELDS");
            if (fieldsPos != std::string::npos) {
                std::string fields = line.substr(fieldsPos + 6);
                columns = splitWhitespace(fields);
                delim = ' ';
            }
            continue;
        }
        if (columns.empty()) {
            if (line.find(',') != std::string::npos) {
                columns = split(line, ',');
                delim = ',';
                for (auto& col : columns) {
                    std::transform(col.begin(), col.end(), col.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                }
                continue;
            } else {
                columns = splitWhitespace(line);
                delim = ' ';
                for (auto& col : columns) {
                    std::transform(col.begin(), col.end(), col.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                }
                continue;
            }
        }
        std::vector<std::string> tokens;
        if (delim == ',') {
            tokens = split(line, ',');
        } else {
            tokens = splitWhitespace(line);
        }
        if (tokens.size() < columns.size()) {
            continue;
        }
        ReferenceRow row;
        for (std::size_t i = 0; i < columns.size(); ++i) {
            std::string col = columns[i];
            std::transform(col.begin(), col.end(), col.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            double value = std::atof(tokens[i].c_str());
            if (col.find("MAX") != std::string::npos) {
                row.maxCluster = value;
            } else if (col.find("NCLUST") != std::string::npos || col.find("NCLUSTERS") != std::string::npos) {
                row.nclusters = value;
            } else if (col.find("TIME") != std::string::npos) {
                row.time = value;
            } else if (col.find("ELAPSED") != std::string::npos) {
                row.elapsedMs = value;
            }
        }
        rows.push_back(row);
    }
    return rows;
}

double kendallTau(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    int concordant = 0;
    int discordant = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = i + 1; j < a.size(); ++j) {
            double diffA = a[i] - a[j];
            double diffB = b[i] - b[j];
            double prod = diffA * diffB;
            if (prod > 0) concordant++;
            else if (prod < 0) discordant++;
        }
    }
    if (concordant + discordant == 0) {
        return 0.0;
    }
    return static_cast<double>(concordant - discordant) / static_cast<double>(concordant + discordant);
}

void writeCsvHeader(std::ostream& os) {
    os << "frame,time_ps,natoms,NX,NY,NZ,dNN_vox,lattice,centering,seeds,seed_hits,nclusters,max_cluster,refined_voxels,elapsed_ms" << std::endl;
}

void writeCsvRow(std::ostream& os, const FrameMetrics& m) {
    os << m.frameIndex << ',' << std::fixed << std::setprecision(3) << m.time_ps << ',' << m.natoms << ','
       << m.nx << ',' << m.ny << ',' << m.nz << ',' << std::setprecision(3) << m.dnn_vox << ','
       << m.lattice << ',' << m.centering << ',' << m.seeds << ',' << m.seedHits << ','
       << m.nclusters << ',' << m.maxCluster << ',' << m.refinedVoxels << ',' << std::setprecision(3) << m.elapsed_ms << std::endl;
}

void writeJsonRow(std::ostream& os, const FrameMetrics& m) {
    os << '{'
       << "\"frame\":" << m.frameIndex << ','
       << "\"time_ps\":" << std::fixed << std::setprecision(3) << m.time_ps << ','
       << "\"natoms\":" << m.natoms << ','
       << "\"NX\":" << m.nx << ','
       << "\"NY\":" << m.ny << ','
       << "\"NZ\":" << m.nz << ','
       << "\"dNN_vox\":" << std::setprecision(3) << m.dnn_vox << ','
       << "\"lattice\":\"" << m.lattice << "\"," << "\"centering\":\"" << m.centering << "\"," << "\"seeds\":" << m.seeds << ','
       << "\"seed_hits\":" << m.seedHits << ','
       << "\"nclusters\":" << m.nclusters << ','
       << "\"max_cluster\":" << m.maxCluster << ','
       << "\"refined_voxels\":" << m.refinedVoxels << ','
       << "\"elapsed_ms\":" << std::setprecision(3) << m.elapsed_ms;
    if (!m.clusterSizes.empty()) {
        os << ",\"cluster_sizes\":[";
        for (std::size_t i = 0; i < m.clusterSizes.size(); ++i) {
            if (i) os << ',';
            os << m.clusterSizes[i];
        }
        os << ']';
    }
    os << '}' << std::endl;
}

}  // namespace bls

int main(int argc, char** argv) {
    using namespace bls;
    CLIOptions options;
    if (!parseCLI(argc, argv, options)) {
        return 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(options.threads);
#endif
    try {
        BLSParameters params = parseConfigFile(options.config);
        if (params.stride > 1) {
            options.stride = std::max(1, options.stride * params.stride);
        }
        BLSAnalyzer analyzer(params);

        Topology topology;
        if (!options.top.empty()) {
            topology = loadTopology(options.top);
        }

        auto reader = createTrajectoryReader(options.traj, options.formatOverride);
        if (!reader) {
            std::cerr << "Unsupported trajectory format." << std::endl;
            return 1;
        }
        if (!reader->open(options.traj)) {
            std::cerr << "Failed to open trajectory: " << options.traj << std::endl;
            return 1;
        }

        std::ofstream csvFile;
        if (!options.outCsv.empty()) {
            csvFile.open(options.outCsv);
            if (!csvFile) {
                std::cerr << "Failed to open CSV output: " << options.outCsv << std::endl;
                return 1;
            }
            writeCsvHeader(csvFile);
        }
        std::ofstream jsonFile;
        if (!options.outJson.empty()) {
            jsonFile.open(options.outJson);
            if (!jsonFile) {
                std::cerr << "Failed to open JSON output: " << options.outJson << std::endl;
                return 1;
            }
        }

        std::vector<FrameMetrics> allMetrics;
        Frame frame;
        int frameIndex = 0;
        int processed = 0;
        while (reader->read(frame)) {
            if (frameIndex < options.start) {
                ++frameIndex;
                continue;
            }
            if (frameIndex > options.stop) {
                break;
            }
            if ((frameIndex - options.start) % options.stride != 0) {
                ++frameIndex;
                continue;
            }
            FrameMetrics metrics = analyzer.analyzeFrame(frameIndex, frame, topology);
            allMetrics.push_back(metrics);
            ++processed;
            logInfo("Frame " + std::to_string(frameIndex) + ": clusters=" + std::to_string(metrics.nclusters) +
                    " max=" + std::to_string(metrics.maxCluster) +
                    " seeds=" + std::to_string(metrics.seeds) +
                    " time_ms=" + std::to_string(metrics.elapsed_ms));
            if (csvFile.is_open()) {
                writeCsvRow(csvFile, metrics);
            }
            if (jsonFile.is_open()) {
                writeJsonRow(jsonFile, metrics);
            }
            ++frameIndex;
        }
        reader->close();

        double totalMs = 0.0;
        for (const auto& m : allMetrics) {
            totalMs += m.elapsed_ms;
        }
        std::cout << "Processed " << processed << " frames. Total time " << totalMs << " ms, average "
                  << (processed > 0 ? totalMs / processed : 0.0) << " ms per frame." << std::endl;
        std::cout << "Peak RSS: " << currentRSS() / (1024.0 * 1024.0) << " MB" << std::endl;

        if (!options.comparePlumed.empty() && !allMetrics.empty()) {
            std::vector<ReferenceRow> reference = loadReference(options.comparePlumed);
            std::size_t pairs = std::min(reference.size(), allMetrics.size());
            if (pairs == 0) {
                std::cout << "No overlapping frames for comparison." << std::endl;
            } else {
                double sumAbsMax = 0.0;
                double sumSqMax = 0.0;
                double sumAbsN = 0.0;
                double sumSqN = 0.0;
                double refTime = 0.0;
                std::vector<double> oursMax;
                std::vector<double> refMax;
                oursMax.reserve(pairs);
                refMax.reserve(pairs);
                for (std::size_t i = 0; i < pairs; ++i) {
                    double diffMax = static_cast<double>(allMetrics[i].maxCluster) - reference[i].maxCluster;
                    double diffN = static_cast<double>(allMetrics[i].nclusters) - reference[i].nclusters;
                    sumAbsMax += std::abs(diffMax);
                    sumSqMax += diffMax * diffMax;
                    sumAbsN += std::abs(diffN);
                    sumSqN += diffN * diffN;
                    refTime += reference[i].elapsedMs;
                    oursMax.push_back(static_cast<double>(allMetrics[i].maxCluster));
                    refMax.push_back(reference[i].maxCluster);
                }
                double meanAbsMax = sumAbsMax / pairs;
                double rmseMax = std::sqrt(sumSqMax / pairs);
                double meanAbsN = sumAbsN / pairs;
                double rmseN = std::sqrt(sumSqN / pairs);
                double tau = kendallTau(oursMax, refMax);
                double oursTime = totalMs;
                double speedup = (oursTime > 0.0 && refTime > 0.0) ? (refTime / oursTime) : 0.0;
                std::cout << "Comparison vs PLUMED:" << std::endl;
                std::cout << "  Mean abs diff max_cluster: " << meanAbsMax << std::endl;
                std::cout << "  RMSE max_cluster: " << rmseMax << std::endl;
                std::cout << "  Mean abs diff nclusters: " << meanAbsN << std::endl;
                std::cout << "  RMSE nclusters: " << rmseN << std::endl;
                std::cout << "  Kendall tau (max_cluster): " << tau << std::endl;
                std::cout << "  Speedup (reference/ours): " << speedup << std::endl;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

