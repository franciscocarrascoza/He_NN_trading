cmake_minimum_required(VERSION 3.15)
project(bls_analyzer LANGUAGES CXX)

option(USE_XDRFILE "Enable XDR trajectory support" OFF)
option(USE_TNG "Enable TNG trajectory support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bls_analyze
    cpp/src/main.cpp
    cpp/src/config/Parser.cpp
    cpp/src/io/TrajectoryFactory.cpp
    cpp/src/io/GroReader.cpp
    cpp/src/io/PdbReader.cpp
    cpp/src/io/XtcReader.cpp
    cpp/src/io/TrrReader.cpp
    cpp/src/io/TngReader.cpp
    cpp/src/lattice/Basis.cpp
    cpp/src/grid/Grid.cpp
    cpp/src/refine/SkipDFS.cpp
    cpp/src/bls/BLS.cpp)

target_include_directories(bls_analyze PRIVATE cpp/src)

target_compile_definitions(bls_analyze PRIVATE $<$<BOOL:${USE_XDRFILE}>:USE_XDRFILE> $<$<BOOL:${USE_TNG}>:USE_TNG>)

if (USE_XDRFILE)
    find_library(XDRFILE_LIBRARY NAMES xdrfile)
    if (XDRFILE_LIBRARY)
        target_link_libraries(bls_analyze PRIVATE ${XDRFILE_LIBRARY})
    else()
        message(WARNING "xdrfile library not found; XTC/TRR support may be unavailable")
    endif()
endif()

if (USE_TNG)
    find_library(TNG_LIBRARY NAMES tng_io)
    if (TNG_LIBRARY)
        target_link_libraries(bls_analyze PRIVATE ${TNG_LIBRARY})
    else()
        message(WARNING "tng_io library not found; TNG support may be unavailable")
    endif()
endif()

enable_testing()
add_executable(bls_tests
    cpp/tests/TestMain.cpp
    cpp/src/config/Parser.cpp
    cpp/src/io/TrajectoryFactory.cpp
    cpp/src/io/GroReader.cpp
    cpp/src/io/PdbReader.cpp
    cpp/src/io/XtcReader.cpp
    cpp/src/io/TrrReader.cpp
    cpp/src/io/TngReader.cpp
    cpp/src/lattice/Basis.cpp
    cpp/src/grid/Grid.cpp
    cpp/src/refine/SkipDFS.cpp
    cpp/src/bls/BLS.cpp)

target_include_directories(bls_tests PRIVATE cpp/src)

target_compile_definitions(bls_tests PRIVATE $<$<BOOL:${USE_XDRFILE}>:USE_XDRFILE> $<$<BOOL:${USE_TNG}>:USE_TNG>)

if (USE_XDRFILE)
    if (XDRFILE_LIBRARY)
        target_link_libraries(bls_tests PRIVATE ${XDRFILE_LIBRARY})
    endif()
endif()
if (USE_TNG)
    if (TNG_LIBRARY)
        target_link_libraries(bls_tests PRIVATE ${TNG_LIBRARY})
    endif()
endif()

add_test(NAME bls_tests COMMAND bls_tests)

