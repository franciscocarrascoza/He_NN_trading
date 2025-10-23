#pragma once

#include <istream>
#include <string>

#include "bls/Parameters.hpp"

namespace bls {

BLSParameters parseConfigFile(const std::string& path);
BLSParameters parseConfigStream(std::istream& input);

}  // namespace bls

