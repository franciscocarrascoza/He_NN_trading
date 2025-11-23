// FIX: Main entry point for He_NN trading desktop application per spec

#include <QApplication>  // FIX: Qt application class
#include "mainwindow.h"  // FIX: main window header

int main(int argc, char *argv[])  // FIX: main function entry point
{
    QApplication app(argc, argv);  // FIX: create Qt application instance

    // FIX: Set application metadata per spec
    QApplication::setApplicationName("He_NN Trading Desktop");  // FIX: application name
    QApplication::setApplicationVersion("1.0.0");  // FIX: application version
    QApplication::setOrganizationName("He_NN");  // FIX: organization name

    // FIX: Create and show main window
    MainWindow window;  // FIX: instantiate main window
    window.setWindowTitle("He_NN Trading - Prediction Desktop");  // FIX: window title per spec
    window.resize(1600, 900);  // FIX: default window size (16:9 aspect ratio)
    window.show();  // FIX: display window

    return app.exec();  // FIX: start event loop
}
