#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace bls {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

class Logger {
  public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void setLevel(LogLevel level) { level_ = level; }

    void log(LogLevel level, const std::string& msg) {
        if (level < level_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%F %T");
        std::clog << "[" << oss.str() << "]" << levelToString(level) << ": " << msg << std::endl;
    }

  private:
    LogLevel level_{LogLevel::Info};
    std::mutex mutex_;

    static const char* levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::Debug:
                return "[DEBUG]";
            case LogLevel::Info:
                return "[INFO ]";
            case LogLevel::Warn:
                return "[WARN ]";
            case LogLevel::Error:
                return "[ERROR]";
        }
        return "[INFO ]";
    }
};

inline void logInfo(const std::string& msg) { Logger::instance().log(LogLevel::Info, msg); }
inline void logWarn(const std::string& msg) { Logger::instance().log(LogLevel::Warn, msg); }
inline void logError(const std::string& msg) { Logger::instance().log(LogLevel::Error, msg); }
inline void logDebug(const std::string& msg) { Logger::instance().log(LogLevel::Debug, msg); }

}  // namespace bls

