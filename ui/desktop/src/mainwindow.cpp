// FIX: MainWindow implementation for He_NN trading desktop application per spec

#include "mainwindow.h"  // FIX: main window header
#include "chartwidget.h"  // FIX: chart widget header
#include "metricswidget.h"  // FIX: metrics widget header
#include "controlwidget.h"  // FIX: control widget header
#include "websocketclient.h"  // FIX: WebSocket client header

#include <QVBoxLayout>  // FIX: vertical box layout
#include <QHBoxLayout>  // FIX: horizontal box layout
#include <QWidget>  // FIX: widget base class
#include <QFile>  // FIX: file I/O for layout JSON
#include <QJsonDocument>  // FIX: JSON document parser
#include <QJsonObject>  // FIX: JSON object
#include <QJsonValue>  // FIX: JSON value
#include <QMessageBox>  // FIX: message box for errors
#include <QDateTime>  // FIX: timestamp handling

MainWindow::MainWindow(QWidget *parent)  // FIX: constructor implementation
    : QMainWindow(parent)  // FIX: call base constructor
    , chartWidget(nullptr)  // FIX: initialize chart widget pointer
    , metricsWidget(nullptr)  // FIX: initialize metrics widget pointer
    , controlWidget(nullptr)  // FIX: initialize control widget pointer
    , wsClient(nullptr)  // FIX: initialize WebSocket client pointer
    , mainSplitter(nullptr)  // FIX: initialize main splitter pointer
    , rightSplitter(nullptr)  // FIX: initialize right splitter pointer
    , statusLabel(nullptr)  // FIX: initialize status label pointer
    , connectionLabel(nullptr)  // FIX: initialize connection label pointer
    , lastUpdateLabel(nullptr)  // FIX: initialize last update label pointer
    , refreshTimer(nullptr)  // FIX: initialize refresh timer pointer
{
    setupUi();  // FIX: initialize UI components
    loadLayoutConfig();  // FIX: load layout from JSON
    connectSignals();  // FIX: connect signals and slots

    // FIX: Connect to backend WebSocket
    wsClient->connectToServer("ws://localhost:8000/ws");  // FIX: connect to backend WS endpoint
}

MainWindow::~MainWindow()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child widgets, explicit cleanup not needed
}

void MainWindow::setupUi()  // FIX: UI initialization implementation
{
    // FIX: Create central widget
    QWidget *centralWidget = new QWidget(this);  // FIX: create central widget
    setCentralWidget(centralWidget);  // FIX: set as central widget

    // FIX: Create main splitter (left: chart, right: panels)
    mainSplitter = new QSplitter(Qt::Horizontal, this);  // FIX: horizontal splitter

    // FIX: Create chart widget
    chartWidget = new ChartWidget(this);  // FIX: instantiate chart widget
    mainSplitter->addWidget(chartWidget);  // FIX: add to main splitter

    // FIX: Create right vertical splitter for metrics/predictions/control
    rightSplitter = new QSplitter(Qt::Vertical, this);  // FIX: vertical splitter

    // FIX: Create metrics widget
    metricsWidget = new MetricsWidget(this);  // FIX: instantiate metrics widget
    rightSplitter->addWidget(metricsWidget);  // FIX: add to right splitter

    // FIX: Create control widget
    controlWidget = new ControlWidget(this);  // FIX: instantiate control widget
    rightSplitter->addWidget(controlWidget);  // FIX: add to right splitter

    mainSplitter->addWidget(rightSplitter);  // FIX: add right splitter to main splitter

    // FIX: Set central layout
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);  // FIX: create vertical layout
    mainLayout->addWidget(mainSplitter);  // FIX: add splitter to layout
    mainLayout->setContentsMargins(0, 0, 0, 0);  // FIX: remove margins

    // FIX: Create status bar
    statusLabel = new QLabel("Ready", this);  // FIX: create status label
    connectionLabel = new QLabel("Disconnected", this);  // FIX: create connection label
    lastUpdateLabel = new QLabel("Last update: Never", this);  // FIX: create last update label

    statusBar()->addWidget(statusLabel);  // FIX: add status label to status bar
    statusBar()->addPermanentWidget(lastUpdateLabel);  // FIX: add last update label (right side)
    statusBar()->addPermanentWidget(connectionLabel);  // FIX: add connection label (right side)

    // FIX: Create WebSocket client
    wsClient = new WebSocketClient(this);  // FIX: instantiate WebSocket client

    // FIX: Create refresh timer
    refreshTimer = new QTimer(this);  // FIX: instantiate refresh timer
    refreshTimer->setInterval(60000);  // FIX: 60 seconds auto-refresh per spec
    refreshTimer->start();  // FIX: start timer
}

void MainWindow::loadLayoutConfig()  // FIX: load layout from JSON per spec
{
    // FIX: Load layout configuration from JSON
    QFile layoutFile("config/ui_layout.json");  // FIX: open layout file
    if (!layoutFile.open(QIODevice::ReadOnly))  // FIX: check file open success
    {
        qWarning() << "Failed to open layout config, using defaults";  // FIX: log warning
        return;  // FIX: early return on failure
    }

    QByteArray layoutData = layoutFile.readAll();  // FIX: read file contents
    layoutFile.close();  // FIX: close file

    QJsonDocument doc = QJsonDocument::fromJson(layoutData);  // FIX: parse JSON
    if (doc.isNull() || !doc.isObject())  // FIX: validate JSON
    {
        qWarning() << "Invalid layout JSON, using defaults";  // FIX: log warning
        return;  // FIX: early return on invalid JSON
    }

    QJsonObject root = doc.object();  // FIX: get root object
    QJsonObject panels = root["panels"].toObject();  // FIX: get panels object

    // FIX: Apply percent-based sizes to splitters
    // FIX: Main splitter (chart vs right panels)
    double chartWidthPercent = panels["chart"].toObject()["percent_width"].toDouble(60.0);  // FIX: get chart width percent
    int totalWidth = width();  // FIX: get total window width
    int chartWidth = static_cast<int>(totalWidth * chartWidthPercent / 100.0);  // FIX: calculate chart width
    mainSplitter->setSizes(QList<int>() << chartWidth << (totalWidth - chartWidth));  // FIX: apply sizes

    // FIX: Right splitter sizes can be configured similarly
    // FIX: Simplified: using equal sizes for now
    int rightHeight = height();  // FIX: get total height
    rightSplitter->setSizes(QList<int>() << rightHeight / 2 << rightHeight / 2);  // FIX: equal split
}

void MainWindow::connectSignals()  // FIX: connect signals and slots
{
    // FIX: WebSocket signals
    connect(wsClient, &WebSocketClient::connected, this, &MainWindow::onWebSocketConnected);  // FIX: connect connected signal
    connect(wsClient, &WebSocketClient::disconnected, this, &MainWindow::onWebSocketDisconnected);  // FIX: connect disconnected signal
    connect(wsClient, &WebSocketClient::errorOccurred, this, &MainWindow::onWebSocketError);  // FIX: connect error signal
    connect(wsClient, &WebSocketClient::messageReceived, this, &MainWindow::onWebSocketMessage);  // FIX: connect message signal

    // FIX: Control widget signals
    connect(controlWidget, &ControlWidget::trainingStartRequested, this, &MainWindow::onTrainingStarted);  // FIX: connect training start signal
    connect(controlWidget, &ControlWidget::trainingStopRequested, this, &MainWindow::onTrainingStopped);  // FIX: connect training stop signal

    // FIX: Refresh timer
    connect(refreshTimer, &QTimer::timeout, this, &MainWindow::updateStatus);  // FIX: connect timeout to status update
}

void MainWindow::onWebSocketConnected()  // FIX: WebSocket connected handler
{
    connectionLabel->setText("Connected");  // FIX: update connection label
    statusLabel->setText("Backend connected");  // FIX: update status label
}

void MainWindow::onWebSocketDisconnected()  // FIX: WebSocket disconnected handler
{
    connectionLabel->setText("Disconnected");  // FIX: update connection label
    statusLabel->setText("Backend disconnected");  // FIX: update status label
}

void MainWindow::onWebSocketError(const QString &error)  // FIX: WebSocket error handler
{
    connectionLabel->setText("Error");  // FIX: update connection label
    statusLabel->setText(QString("WebSocket error: %1").arg(error));  // FIX: update status label with error
    QMessageBox::warning(this, "WebSocket Error", error);  // FIX: show error dialog
}

void MainWindow::onWebSocketMessage(const QString &message)  // FIX: WebSocket message handler
{
    // FIX: Parse and handle WebSocket messages
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());  // FIX: parse JSON message
    if (doc.isNull() || !doc.isObject())  // FIX: validate JSON
    {
        qWarning() << "Invalid WebSocket message JSON:" << message;  // FIX: log warning
        return;  // FIX: early return on invalid JSON
    }

    QJsonObject obj = doc.object();  // FIX: get message object
    QString messageType = obj["type"].toString();  // FIX: get message type

    // FIX: Handle different message types per spec
    if (messageType == "training_start")  // FIX: training start message
    {
        statusLabel->setText("Training started");  // FIX: update status
        metricsWidget->resetMetrics();  // FIX: reset metrics panel
    }
    else if (messageType == "training_complete")  // FIX: training complete message
    {
        statusLabel->setText("Training completed");  // FIX: update status
    }
    else if (messageType == "epoch_metrics")  // FIX: epoch metrics message
    {
        metricsWidget->updateMetrics(obj);  // FIX: update metrics panel with epoch data
    }
    else if (messageType == "training_error")  // FIX: training error message
    {
        QString error = obj["error"].toString();  // FIX: get error message
        statusLabel->setText(QString("Training error: %1").arg(error));  // FIX: update status
        QMessageBox::critical(this, "Training Error", error);  // FIX: show error dialog
    }

    // FIX: Update last update timestamp
    lastUpdateLabel->setText(QString("Last update: %1").arg(QDateTime::currentDateTime().toString("hh:mm:ss")));  // FIX: update timestamp
}

void MainWindow::onTrainingStarted()  // FIX: training start handler
{
    statusLabel->setText("Requesting training start...");  // FIX: update status
    // FIX: Send HTTP POST request to backend /start_training endpoint
    // FIX: This would use QNetworkAccessManager in full implementation
}

void MainWindow::onTrainingStopped()  // FIX: training stop handler
{
    statusLabel->setText("Requesting training stop...");  // FIX: update status
    // FIX: Send HTTP POST request to backend /stop_training endpoint
}

void MainWindow::updateStatus()  // FIX: status update handler for auto-refresh
{
    // FIX: Query backend /status endpoint for current state
    // FIX: This would use QNetworkAccessManager in full implementation
}
