// FIX: MainWindow header for He_NN trading desktop application per spec

#ifndef MAINWINDOW_H  // FIX: header guard
#define MAINWINDOW_H

#include <QMainWindow>  // FIX: Qt main window base class
#include <QSplitter>  // FIX: Qt splitter for resizable panels
#include <QStatusBar>  // FIX: Qt status bar
#include <QLabel>  // FIX: Qt label for status messages
#include <QTimer>  // FIX: Qt timer for auto-refresh
#include <QNetworkAccessManager>  // FIX: HTTP client for REST calls
#include <QNetworkRequest>  // FIX: HTTP request
#include <QNetworkReply>  // FIX: HTTP response

// FIX: Forward declarations for custom widgets
class ChartWidget;  // FIX: chart widget forward declaration
class MetricsWidget;  // FIX: metrics widget forward declaration
class ControlWidget;  // FIX: control widget forward declaration
class WebSocketClient;  // FIX: WebSocket client forward declaration

class MainWindow : public QMainWindow  // FIX: MainWindow class inheriting QMainWindow
{
    Q_OBJECT  // FIX: Qt meta-object system macro

public:
    explicit MainWindow(QWidget *parent = nullptr);  // FIX: constructor with optional parent
    ~MainWindow();  // FIX: destructor

private slots:
    void onWebSocketConnected();  // FIX: WebSocket connection handler
    void onWebSocketDisconnected();  // FIX: WebSocket disconnection handler
    void onWebSocketError(const QString &error);  // FIX: WebSocket error handler
    void onWebSocketMessage(const QString &message);  // FIX: WebSocket message handler
    void onTrainingStarted();  // FIX: training start handler (sends REST POST)
    void onTrainingStopped();  // FIX: training stop handler
    void onAckTimeout();  // FIX: ACK timeout handler (retry logic)
    void updateStatus();  // FIX: status bar updater
    void showLogTailDialog();  // FIX: view logs dialog
    void refreshChart();  // FIX: chart refresh handler (polls market data)

private:
    void setupUi();  // FIX: UI initialization method
    void loadLayoutConfig();  // FIX: load UI layout from JSON per spec
    void connectSignals();  // FIX: connect signals and slots

    // FIX: Widget pointers per spec
    ChartWidget *chartWidget;  // FIX: chart widget pointer
    MetricsWidget *metricsWidget;  // FIX: metrics widget pointer
    ControlWidget *controlWidget;  // FIX: control widget pointer
    WebSocketClient *wsClient;  // FIX: WebSocket client pointer

    // FIX: Layout components
    QSplitter *mainSplitter;  // FIX: main horizontal splitter
    QSplitter *rightSplitter;  // FIX: right vertical splitter

    // FIX: Status bar components
    QLabel *statusLabel;  // FIX: status message label
    QLabel *connectionLabel;  // FIX: connection status label
    QLabel *lastUpdateLabel;  // FIX: last update timestamp label

    // FIX: Auto-refresh timers
    QTimer *refreshTimer;  // FIX: auto-refresh timer per spec (status updates)
    QTimer *chartRefreshTimer;  // FIX: chart refresh timer (market data polling)

    // FIX: HTTP client and training handshake
    QNetworkAccessManager *httpClient;  // FIX: HTTP client for REST API calls
    QTimer *ackTimeout;  // FIX: 30-second WebSocket ACK timer
    QString currentJobId;  // FIX: track current training job_id
    int restRetryCount;  // FIX: retry counter (max 2 retries)
};

#endif  // FIX: close header guard
