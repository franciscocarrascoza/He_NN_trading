# FIX: CMake build configuration for Qt6 desktop application per spec
cmake_minimum_required(VERSION 3.16)  # FIX: minimum CMake version for Qt6

project(HeNNTradingDesktop VERSION 1.0 LANGUAGES CXX)  # FIX: project name and version

set(CMAKE_CXX_STANDARD 17)  # FIX: C++17 standard per spec
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # FIX: require standard
set(CMAKE_AUTOMOC ON)  # FIX: enable MOC for Qt
set(CMAKE_AUTORCC ON)  # FIX: enable RCC for Qt resources
set(CMAKE_AUTOUIC ON)  # FIX: enable UIC for Qt UI files

# FIX: Find Qt6 packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network WebSockets Charts)  # FIX: Qt6 dependencies per spec

# FIX: Source files
set(SOURCES
    src/main.cpp  # FIX: main entry point
    src/mainwindow.cpp  # FIX: main window implementation
    src/chartwidget.cpp  # FIX: chart widget for OHLCV + overlays
    src/metricswidget.cpp  # FIX: metrics panel
    src/controlwidget.cpp  # FIX: control panel
    src/websocketclient.cpp  # FIX: WebSocket client for backend streaming
)  # FIX: source file list

# FIX: Header files
set(HEADERS
    src/mainwindow.h  # FIX: main window header
    src/chartwidget.h  # FIX: chart widget header
    src/metricswidget.h  # FIX: metrics panel header
    src/controlwidget.h  # FIX: control panel header
    src/websocketclient.h  # FIX: WebSocket client header
)  # FIX: header file list

# FIX: Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})  # FIX: build executable

# FIX: Link Qt6 libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core  # FIX: Qt core
    Qt6::Widgets  # FIX: Qt widgets
    Qt6::Network  # FIX: Qt network for HTTP
    Qt6::WebSockets  # FIX: Qt WebSockets
    Qt6::Charts  # FIX: Qt Charts for candlestick visualization
)  # FIX: link libraries

# FIX: Install target
install(TARGETS ${PROJECT_NAME}
    BUNDLE DESTINATION .  # FIX: macOS bundle destination
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # FIX: Linux library destination
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # FIX: Windows/Linux runtime destination
)  # FIX: install configuration
