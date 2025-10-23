#include "config/Parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/Logging.hpp"

namespace bls {

namespace {

std::string trim(const std::string& input) {
    auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

std::vector<std::string> splitWhitespace(const std::string& input) {
    std::istringstream iss(input);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> split(const std::string& input, char delim) {
    std::vector<std::string> result;
    std::string token;
    std::istringstream iss(input);
    while (std::getline(iss, token, delim)) {
        token = trim(token);
        if (!token.empty()) {
            result.push_back(token);
        }
    }
    return result;
}

std::string toUpper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

void parseGroup(const std::string& spec, GroupSpec& group) {
    group = GroupSpec{};
    auto parts = split(spec, '|');
    for (auto& rawPart : parts) {
        std::string part = trim(rawPart);
        std::string upperPart = toUpper(part);
        if (upperPart.find("ATOMS=ALL") != std::string::npos) {
            group.all = true;
            continue;
        }
        auto colon = part.find(':');
        if (colon != std::string::npos) {
            std::string key = toUpper(trim(part.substr(0, colon)));
            std::string value = trim(part.substr(colon + 1));
            value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
            if (key == "INDEX") {
                group.all = false;
                auto idxTokens = split(value, ',');
                for (const auto& token : idxTokens) {
                    auto dash = token.find('-');
                    if (dash != std::string::npos) {
                        int start = std::stoi(token.substr(0, dash));
                        int end = std::stoi(token.substr(dash + 1));
                        if (end < start) {
                            std::swap(start, end);
                        }
                        for (int i = start; i <= end; ++i) {
                            group.indices.push_back(i - 1);
                        }
                    } else {
                        group.indices.push_back(std::stoi(token) - 1);
                    }
                }
            } else if (key == "NAME") {
                group.all = false;
                group.nameFilter = value;
            }
        }
    }
    if (!group.indices.empty()) {
        std::sort(group.indices.begin(), group.indices.end());
        group.indices.erase(std::unique(group.indices.begin(), group.indices.end()), group.indices.end());
    }
}

void parseBox(const std::vector<std::string>& tokens, BoxSpec& box) {
    box = BoxSpec{};
    std::vector<std::string> filtered;
    filtered.reserve(tokens.size());
    for (const auto& token : tokens) {
        if (token == "|") continue;
        filtered.push_back(token);
    }
    if (filtered.size() == 1 && toUpper(filtered[0]) == "AUTO") {
        box.autoBox = true;
        return;
    }
    box.autoBox = false;
    for (std::size_t i = 0; i + 1 < filtered.size(); i += 2) {
        std::string key = toUpper(filtered[i]);
        double value = std::stod(filtered[i + 1]);
        if (key == "XLO") box.lower.x = value;
        if (key == "YLO") box.lower.y = value;
        if (key == "ZLO") box.lower.z = value;
        if (key == "XHI") box.upper.x = value;
        if (key == "YHI") box.upper.y = value;
        if (key == "ZHI") box.upper.z = value;
    }
}

ConnectivityMode parseConnectivity(const std::string& token) {
    int value = std::stoi(token);
    switch (value) {
        case 6:
            return ConnectivityMode::Six;
        case 18:
            return ConnectivityMode::Eighteen;
        case 26:
            return ConnectivityMode::TwentySix;
        default:
            throw std::runtime_error("Unsupported connectivity: " + token);
    }
}

LatticeType parseLattice(const std::string& token) {
    std::string upper = toUpper(token);
    if (upper == "CUBIC") return LatticeType::Cubic;
    if (upper == "HEXAGONAL") return LatticeType::Hexagonal;
    if (upper == "TRICLINIC") return LatticeType::Triclinic;
    throw std::runtime_error("Unsupported lattice: " + token);
}

CenteringType parseCentering(const std::string& token) {
    std::string upper = toUpper(token);
    if (upper == "P") return CenteringType::P;
    if (upper == "F") return CenteringType::F;
    if (upper == "I") return CenteringType::I;
    throw std::runtime_error("Unsupported centering: " + token);
}

OccupancyRule parseOccupancy(const std::string& token) {
    std::string upper = toUpper(token);
    if (upper == "ANY") return OccupancyRule::Any;
    if (upper == "ALL") return OccupancyRule::All;
    throw std::runtime_error("Unsupported occupancy rule: " + token);
}

}  // namespace

BLSParameters parseConfigStream(std::istream& input) {
    BLSParameters params;
    std::string line;
    bool inBlock = false;
    while (std::getline(input, line)) {
        auto commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (!inBlock) {
            std::string upper = toUpper(line);
            if (upper.rfind("BLS", 0) == 0) {
                inBlock = true;
            }
            continue;
        }
        if (line == "... BLS" || line == "...BLS") {
            break;
        }
        auto tokens = splitWhitespace(line);
        if (tokens.empty()) {
            continue;
        }
        std::string key = toUpper(tokens[0]);
        tokens.erase(tokens.begin());
        if (key == "GROUP") {
            std::string rest = line.substr(line.find_first_of(" ") + 1);
            parseGroup(rest, params.group);
        } else if (key == "BOX") {
            parseBox(tokens, params.box);
        } else if (key == "GRID_SPACING") {
            if (!tokens.empty()) params.gridSpacing = std::stod(tokens[0]);
        } else if (key == "CONNECTIVITY") {
            if (!tokens.empty()) params.connectivity = parseConnectivity(tokens[0]);
        } else if (key == "ALPHA") {
            if (!tokens.empty()) params.alpha = std::stod(tokens[0]);
        } else if (key == "RADII") {
            params.radii.clear();
            if (!tokens.empty()) {
                auto values = split(tokens[0], ',');
                for (const auto& v : values) {
                    params.radii.push_back(std::stod(v));
                }
            }
        } else if (key == "DNN") {
            if (!tokens.empty()) params.dnn = std::stod(tokens[0]);
        } else if (key == "LATTICE") {
            if (!tokens.empty()) params.lattice = parseLattice(tokens[0]);
        } else if (key == "CENTERING") {
            if (!tokens.empty()) params.centering = parseCentering(tokens[0]);
        } else if (key == "HEX_C_OVER_A") {
            if (!tokens.empty()) params.hex_c_over_a = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_A") {
            if (!tokens.empty()) params.triclinic_abc.x = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_B") {
            if (!tokens.empty()) params.triclinic_abc.y = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_C") {
            if (!tokens.empty()) params.triclinic_abc.z = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_ALPHA") {
            if (!tokens.empty()) params.triclinic_angles.x = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_BETA") {
            if (!tokens.empty()) params.triclinic_angles.y = std::stod(tokens[0]);
        } else if (key == "TRICLINIC_GAMMA") {
            if (!tokens.empty()) params.triclinic_angles.z = std::stod(tokens[0]);
        } else if (key == "SKIP") {
            if (!tokens.empty()) params.skip = std::stoi(tokens[0]);
        } else if (key == "STRIDE") {
            if (!tokens.empty()) params.stride = std::stoi(tokens[0]);
        } else if (key == "OCCUPANCY") {
            if (!tokens.empty()) params.occupancy = parseOccupancy(tokens[0]);
        } else if (key == "CUTOFF") {
            if (!tokens.empty()) params.cutoff = std::stod(tokens[0]);
        } else if (key == "OUTPUT") {
            params.outputs.clear();
            if (!tokens.empty()) {
                auto values = split(tokens[0], ',');
                for (auto& v : values) {
                    params.outputs.push_back(toUpper(v));
                }
            }
        }
    }
    return params;
}

BLSParameters parseConfigFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open configuration file: " + path);
    }
    return parseConfigStream(file);
}

}  // namespace bls

