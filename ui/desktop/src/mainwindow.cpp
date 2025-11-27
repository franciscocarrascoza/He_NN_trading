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
#include <QJsonArray>  // FIX: JSON array
#include <QMessageBox>  // FIX: message box for errors
#include <QDateTime>  // FIX: timestamp handling
#include <QTextEdit>  // FIX: text edit for log viewer
#include <QDialog>  // FIX: dialog for log viewer
#include <QPushButton>  // FIX: push button
#include <QFont>  // FIX: font for log viewer

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
    , chartRefreshTimer(nullptr)  // FIX: initialize chart refresh timer pointer
    , httpClient(nullptr)  // FIX: initialize HTTP client pointer
    , ackTimeout(nullptr)  // FIX: initialize ACK timeout timer pointer
    , currentJobId("")  // FIX: initialize job ID to empty string
    , restRetryCount(0)  // FIX: initialize retry counter to 0
{
    setupUi();  // FIX: initialize UI components
    loadLayoutConfig();  // FIX: load layout from JSON
    connectSignals();  // FIX: connect signals and slots

    // FIX: Initialize HTTP client for REST API calls
    httpClient = new QNetworkAccessManager(this);  // FIX: create HTTP client

    // FIX: Initialize ACK timeout timer
    ackTimeout = new QTimer(this);  // FIX: create timeout timer
    ackTimeout->setSingleShot(true);  // FIX: fire only once per start
    connect(ackTimeout, &QTimer::timeout, this, &MainWindow::onAckTimeout);  // FIX: connect timeout handler

    // FIX: Initialize chart refresh timer (60-second polling)
    chartRefreshTimer = new QTimer(this);  // FIX: create chart refresh timer
    chartRefreshTimer->setInterval(60000);  // FIX: 60 seconds per spec
    connect(chartRefreshTimer, &QTimer::timeout, this, &MainWindow::refreshChart);  // FIX: connect refresh handler
    chartRefreshTimer->start();  // FIX: start polling immediately

    // FIX: Do initial chart refresh
    refreshChart();  // FIX: load initial data

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

void MainWindow::loadLayoutConfig()  // FIX: load layout from JSON per spec with robust fallback
{
    QString layoutPath = "config/ui_layout.json";  // FIX: layout file path
    QFile layoutFile(layoutPath);  // FIX: open layout file

    bool needsBackup = false;  // FIX: track if backup is needed
    bool needsDefault = false;  // FIX: track if default file creation is needed

    // FIX: Try to open and parse layout file
    if (!layoutFile.open(QIODevice::ReadOnly))  // FIX: check file open success
    {
        qWarning() << "Layout config not found at" << layoutPath << "- will create default";  // FIX: log warning
        needsDefault = true;  // FIX: flag for default creation
    }
    else
    {
        QByteArray layoutData = layoutFile.readAll();  // FIX: read file contents
        layoutFile.close();  // FIX: close file

        QJsonDocument doc = QJsonDocument::fromJson(layoutData);  // FIX: parse JSON
        if (doc.isNull() || !doc.isObject())  // FIX: validate JSON
        {
            qWarning() << "Layout config JSON is invalid/corrupted - backing up and creating default";  // FIX: log warning
            needsBackup = true;  // FIX: flag for backup
            needsDefault = true;  // FIX: flag for default creation
        }
        else
        {
            // FIX: Valid JSON, apply layout
            QJsonObject root = doc.object();  // FIX: get root object
            QJsonObject panels = root["panels"].toObject();  // FIX: get panels object

            // FIX: Apply percent-based sizes to splitters
            double chartWidthPercent = panels["chart"].toObject()["percent_width"].toDouble(60.0);  // FIX: chart width percent
            int totalWidth = width();  // FIX: get total window width
            int chartWidth = static_cast<int>(totalWidth * chartWidthPercent / 100.0);  // FIX: calculate chart width
            mainSplitter->setSizes(QList<int>() << chartWidth << (totalWidth - chartWidth));  // FIX: apply sizes

            // FIX: Right splitter sizes
            int rightHeight = height();  // FIX: get total height
            rightSplitter->setSizes(QList<int>() << rightHeight / 2 << rightHeight / 2);  // FIX: equal split

            qInfo() << "Layout config loaded successfully from" << layoutPath;  // FIX: log success
            return;  // FIX: early return on success
        }
    }

    // FIX: Backup corrupted file if needed
    if (needsBackup)
    {
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");  // FIX: timestamp
        QString backupPath = QString("config/ui_layout.json.bak.%1").arg(timestamp);  // FIX: backup path

        if (QFile::copy(layoutPath, backupPath))  // FIX: create backup
        {
            qInfo() << "Backed up corrupted layout to" << backupPath;  // FIX: log backup success
        }
        else
        {
            qWarning() << "Failed to backup corrupted layout file";  // FIX: log backup failure
        }
    }

    // FIX: Create default layout file if needed
    if (needsDefault)
    {
        QString defaultPath = "config/ui_layout.json.default";  // FIX: default file path

        // FIX: Default layout JSON with percent-based dimensions
        QJsonObject defaultLayout;
        defaultLayout["_comment"] = "FIX: UI layout configuration with percent-based dimensions per spec";
        defaultLayout["version"] = "1.0";

        // FIX: Define default panel layout
        QJsonObject panels;

        QJsonObject chart;
        chart["percent_width"] = 50.0;  // FIX: increased chart width to 50%
        chart["percent_height"] = 70.0;
        chart["position"] = "left";
        panels["chart"] = chart;

        QJsonObject metrics;
        metrics["percent_width"] = 50.0;  // FIX: metrics panel width 50% per user spec
        metrics["percent_height"] = 50.0;
        metrics["position"] = "right-top";
        panels["metrics"] = metrics;

        QJsonObject control;
        control["percent_width"] = 50.0;  // FIX: control panel width 50%
        control["percent_height"] = 50.0;
        control["position"] = "right-bottom";
        panels["control"] = control;

        defaultLayout["panels"] = panels;

        // FIX: Write default file
        QFile defaultFile(defaultPath);
        if (defaultFile.open(QIODevice::WriteOnly))  // FIX: open for writing
        {
            QJsonDocument defaultDoc(defaultLayout);  // FIX: create JSON document
            defaultFile.write(defaultDoc.toJson(QJsonDocument::Indented));  // FIX: write formatted JSON
            defaultFile.close();  // FIX: close file
            qInfo() << "Created default layout at" << defaultPath;  // FIX: log creation

            // FIX: Copy default to main config path
            if (QFile::copy(defaultPath, layoutPath))
            {
                qInfo() << "Copied default layout to" << layoutPath;  // FIX: log copy
            }
        }

        // FIX: Show non-blocking warning to user
        QTimer::singleShot(1000, this, [this]() {  // FIX: delay to let GUI render
            QMessageBox msgBox(this);
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.setWindowTitle("Layout Config Reset");
            msgBox.setText("Layout configuration was missing or corrupted.");
            msgBox.setInformativeText("Using default layout. You can customize it in Settings → Layout Editor.");
            msgBox.setStandardButtons(QMessageBox::Ok);
            msgBox.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
            msgBox.exec();
        });
    }
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
    // FIX: Handle training.started event (compatibility with both "training.started" and "training_start")
    if (messageType == "training.started" || messageType == "training_start")  // FIX: training start confirmation
    {
        QString jobId = obj["job_id"].toString();  // FIX: get job_id from event

        // FIX: Verify job_id matches current request (or accept if no job_id tracked yet)
        if (jobId == currentJobId || currentJobId.isEmpty()) {
            // FIX: Stop ACK timeout timer (training confirmed!)
            ackTimeout->stop();  // FIX: cancel timeout

            // FIX: Update GUI state to "Training running"
            statusLabel->setText("Training running");  // FIX: update status label
            connectionLabel->setText("✓ Connected | Training active");  // FIX: update connection status

            // FIX: Reset metrics panel for new training session
            metricsWidget->resetMetrics();  // FIX: reset metrics

            qInfo() << "Training started confirmed via WebSocket, job_id:" << jobId;  // FIX: log confirmation
        } else {
            qWarning() << "Received training.started for different job_id:" << jobId << "(expected:" << currentJobId << ")";  // FIX: log mismatch
        }
    }
    // FIX: Handle training.failed event
    else if (messageType == "training.failed" || messageType == "training_error")  // FIX: training failure
    {
        ackTimeout->stop();  // FIX: stop ACK timer
        QString reason = obj["reason"].toString();  // FIX: get failure reason
        if (reason.isEmpty()) {
            reason = obj["error"].toString();  // FIX: fallback to error field
        }
        statusLabel->setText(QString("Training failed: %1").arg(reason));  // FIX: update status
        QMessageBox::critical(this, "Training Failed", reason);  // FIX: show error dialog
    }
    else if (messageType == "training_complete")  // FIX: training complete message
    {
        statusLabel->setText("Training completed");  // FIX: update status
    }
    else if (messageType == "epoch_metrics")  // FIX: epoch metrics message
    {
        metricsWidget->updateMetrics(obj);  // FIX: update metrics panel with epoch data
    }
    // FIX: Handle prediction.update events (for chart overlay)
    else if (messageType == "prediction.update")  // FIX: prediction update event
    {
        // FIX: Extract prediction data
        int timestamp = obj["timestamp"].toInt();  // FIX: get timestamp
        double mu = obj["mu"].toDouble();  // FIX: get predicted mean
        double conformalLower = obj["conformal_lower"].toDouble();  // FIX: get lower bound
        double conformalUpper = obj["conformal_upper"].toDouble();  // FIX: get upper bound

        // FIX: TODO: Add prediction overlay to chart widget
        // chartWidget->addPredictionOverlay(timestamp, mu, conformalLower, conformalUpper);

        qInfo() << "Received prediction:" << timestamp << mu << conformalLower << conformalUpper;  // FIX: log prediction
    }

    // FIX: Update last update timestamp
    lastUpdateLabel->setText(QString("Last update: %1").arg(QDateTime::currentDateTime().toString("hh:mm:ss")));  // FIX: update timestamp
}

void MainWindow::onTrainingStarted()  // FIX: training start handler with full REST POST
{
    // FIX: Collect all parameters from control panel
    QJsonObject requestBody = controlWidget->getAllParameters();  // FIX: get all 30+ parameters

    // FIX: Client-side validation
    if (requestBody["symbol"].toString().isEmpty()) {
        QMessageBox::warning(this, "Validation Error", "Symbol cannot be empty");
        return;
    }
    if (requestBody["batch_size"].toInt() <= 0) {
        QMessageBox::warning(this, "Validation Error", "Batch size must be positive");
        return;
    }

    // FIX: Update status and disable buttons
    statusLabel->setText("Requesting training start...");  // FIX: update status label

    // FIX: Create HTTP POST request
    QNetworkRequest request(QUrl("http://localhost:8000/start_training"));  // FIX: backend endpoint
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");  // FIX: JSON content type

    // FIX: Send POST request with JSON body
    QNetworkReply *reply = httpClient->post(request, QJsonDocument(requestBody).toJson());  // FIX: send request

    // FIX: Handle REST response asynchronously
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();  // FIX: auto-delete reply object

        if (reply->error() != QNetworkReply::NoError) {
            // FIX: Network error occurred
            QString errorMsg = reply->errorString();  // FIX: get error message
            statusLabel->setText(QString("Error: %1").arg(errorMsg));  // FIX: update status
            QMessageBox::critical(this, "Training Start Failed",
                QString("Network error: %1").arg(errorMsg));  // FIX: show error dialog
            return;
        }

        // FIX: Parse JSON response
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());  // FIX: parse response body
        if (doc.isNull() || !doc.isObject()) {
            statusLabel->setText("Invalid response from backend");  // FIX: update status
            QMessageBox::critical(this, "Training Start Failed", "Invalid JSON response");  // FIX: show error
            return;
        }

        QJsonObject response = doc.object();  // FIX: get response object

        // FIX: Check if training was accepted (HTTP 202)
        if (response["status"].toString() == "accepted") {
            currentJobId = response["job_id"].toString();  // FIX: save job_id for tracking
            statusLabel->setText("Training accepted — awaiting backend confirmation...");  // FIX: update status

            // FIX: Start 30-second ACK timeout timer
            ackTimeout->start(30000);  // FIX: 30 seconds timeout
            restRetryCount = 0;  // FIX: reset retry counter

            qInfo() << "Training request accepted, job_id:" << currentJobId;  // FIX: log success
        } else {
            // FIX: Unexpected response status
            QString msg = response["message"].toString();  // FIX: get message
            statusLabel->setText(QString("Unexpected response: %1").arg(msg));  // FIX: update status
            QMessageBox::warning(this, "Training Start",
                QString("Backend response: %1").arg(msg));  // FIX: show warning
        }
    });
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

void MainWindow::onAckTimeout()  // FIX: ACK timeout handler with retry logic
{
    // FIX: No WebSocket ACK received within 30 seconds
    if (restRetryCount < 2) {
        // FIX: Retry REST call (max 2 retries = 3 total attempts)
        restRetryCount++;  // FIX: increment retry counter
        statusLabel->setText(QString("Retrying training start (attempt %1/3)...").arg(restRetryCount + 1));  // FIX: update status

        qWarning() << "No WebSocket ACK received, retrying..." << restRetryCount;  // FIX: log warning

        // FIX: Re-trigger training start by calling onTrainingStarted again
        onTrainingStarted();  // FIX: retry POST request

    } else {
        // FIX: Max retries exceeded (3 attempts total), show error
        statusLabel->setText("Training start failed: no backend acknowledgement");  // FIX: update status

        // FIX: Create error dialog with action buttons
        QMessageBox msgBox(this);  // FIX: create message box
        msgBox.setIcon(QMessageBox::Critical);  // FIX: critical icon
        msgBox.setWindowTitle("Training Start Failed");  // FIX: window title
        msgBox.setText("Unable to start training: backend did not acknowledge.");  // FIX: main text
        msgBox.setInformativeText("The training request was sent but no confirmation was received from the backend after 3 attempts. Check backend logs for details.");  // FIX: detailed text

        // FIX: Add custom action buttons
        QPushButton *viewLogsBtn = msgBox.addButton("View Logs", QMessageBox::ActionRole);  // FIX: view logs button
        QPushButton *retryBtn = msgBox.addButton("Retry", QMessageBox::ActionRole);  // FIX: retry button
        msgBox.addButton(QMessageBox::Cancel);  // FIX: cancel button

        msgBox.exec();  // FIX: show dialog and wait for user action

        // FIX: Handle button clicks
        if (msgBox.clickedButton() == viewLogsBtn) {
            // FIX: Show log tail dialog
            showLogTailDialog();  // FIX: fetch and display backend logs
        } else if (msgBox.clickedButton() == retryBtn) {
            // FIX: Reset retry counter and try again
            restRetryCount = 0;  // FIX: reset counter
            onTrainingStarted();  // FIX: retry POST request
        }
    }
}

void MainWindow::showLogTailDialog()  // FIX: view logs dialog implementation
{
    // FIX: Fetch logs from backend /status/log_tail endpoint
    QNetworkRequest request(QUrl("http://localhost:8000/status/log_tail?n=200"));  // FIX: backend logs endpoint
    QNetworkReply *reply = httpClient->get(request);  // FIX: send GET request

    // FIX: Handle response asynchronously
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();  // FIX: auto-delete reply object

        if (reply->error() != QNetworkReply::NoError) {
            // FIX: Network error occurred
            QMessageBox::warning(this, "Log Fetch Failed",
                QString("Failed to fetch logs: %1").arg(reply->errorString()));  // FIX: show error
            return;
        }

        // FIX: Parse JSON response
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());  // FIX: parse response
        if (doc.isNull() || !doc.isObject()) {
            QMessageBox::warning(this, "Log Fetch Failed", "Invalid JSON response");  // FIX: show error
            return;
        }

        QJsonObject response = doc.object();  // FIX: get response object
        QJsonArray lines = response["lines"].toArray();  // FIX: get log lines array

        // FIX: Build log text from lines
        QString logText;  // FIX: log text string
        for (const QJsonValue &line : lines) {
            logText += line.toString() + "\n";  // FIX: append each line
        }

        // FIX: Create dialog to display logs
        QDialog *logDialog = new QDialog(this);  // FIX: create dialog
        logDialog->setWindowTitle("Backend Logs (Last 200 lines)");  // FIX: dialog title
        logDialog->resize(800, 600);  // FIX: dialog size

        // FIX: Create text edit widget for logs
        QTextEdit *textEdit = new QTextEdit(logDialog);  // FIX: create text edit
        textEdit->setReadOnly(true);  // FIX: read-only mode
        textEdit->setPlainText(logText);  // FIX: set log content
        textEdit->setFont(QFont("Courier", 9));  // FIX: monospace font

        // FIX: Create dialog layout
        QVBoxLayout *layout = new QVBoxLayout(logDialog);  // FIX: create layout
        layout->addWidget(textEdit);  // FIX: add text edit

        // FIX: Add close button
        QPushButton *closeBtn = new QPushButton("Close", logDialog);  // FIX: create close button
        connect(closeBtn, &QPushButton::clicked, logDialog, &QDialog::accept);  // FIX: connect to close
        layout->addWidget(closeBtn);  // FIX: add button to layout

        logDialog->exec();  // FIX: show dialog modally
    });
}

void MainWindow::refreshChart()  // FIX: chart refresh handler (polls market data)
{
    // FIX: Get current symbol and timeframe from control panel
    QJsonObject params = controlWidget->getAllParameters();  // FIX: get all parameters
    QString symbol = params["symbol"].toString();  // FIX: extract symbol
    QString timeframe = params["timeframe"].toString();  // FIX: extract timeframe

    // FIX: Build URL with query parameters
    QString url = QString("http://localhost:8000/market_data/latest?symbol=%1&timeframe=%2&limit=100")
        .arg(symbol, timeframe);  // FIX: format URL with parameters

    // FIX: Send GET request to backend
    QNetworkRequest request{QUrl(url)};  // FIX: create request (using braces to avoid vexing parse)
    QNetworkReply *reply = httpClient->get(request);  // FIX: send GET request

    // FIX: Handle response asynchronously
    connect(reply, &QNetworkReply::finished, this, [this, reply, symbol, timeframe]() {
        reply->deleteLater();  // FIX: auto-delete reply object

        if (reply->error() != QNetworkReply::NoError) {
            // FIX: Network error occurred (silently fail, just log)
            qWarning() << "Chart refresh failed:" << reply->errorString();  // FIX: log error
            return;  // FIX: don't show error dialog, just skip this refresh
        }

        // FIX: Parse JSON response
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());  // FIX: parse response
        if (doc.isNull() || !doc.isObject()) {
            qWarning() << "Chart refresh: Invalid JSON response";  // FIX: log error
            return;  // FIX: skip this refresh
        }

        QJsonObject response = doc.object();  // FIX: get response object
        QJsonArray candles = response["candles"].toArray();  // FIX: get candles array

        if (candles.isEmpty()) {
            qWarning() << "Chart refresh: No candles in response";  // FIX: log warning
            return;  // FIX: skip this refresh
        }

        // FIX: Build complete data object for chart update
        QJsonObject chartData;  // FIX: create data object
        chartData["symbol"] = symbol;  // FIX: add symbol
        chartData["timeframe"] = timeframe;  // FIX: add timeframe
        chartData["candles"] = candles;  // FIX: add candles array

        // FIX: Update chart widget with new data
        chartWidget->updateChart(chartData);  // FIX: full chart replace

        // FIX: Update last update label
        lastUpdateLabel->setText(QString("Last update: %1").arg(QDateTime::currentDateTime().toString("hh:mm:ss")));  // FIX: update timestamp

        qInfo() << "Chart refreshed:" << candles.size() << "candles for" << symbol << timeframe;  // FIX: log success
    });
}
