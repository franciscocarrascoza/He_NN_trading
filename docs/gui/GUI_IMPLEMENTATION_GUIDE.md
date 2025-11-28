# GUI Implementation Guide - Missing Features

## Overview
The backend is now fully functional with REST + WebSocket handshake support. The GUI needs updates to utilize these features.

---

## 🔴 CRITICAL: Missing GUI Controls

### Current Control Panel (`ui/desktop/src/controlwidget.cpp`)
**Has:**
- ✅ Timeframe dropdown (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- ✅ Forecast horizon spinner (1-100)
- ✅ Calibration method dropdown (abs, std_gauss)
- ✅ p_up source dropdown (cdf, logit)
- ✅ CV enable checkbox

**Missing:**
- ❌ Symbol text input (default: "BTCUSDT")
- ❌ Batch size spinner (default: 512, range: 1-2048)
- ❌ CV folds spinner (default: 5, range: 1-20)

### How to Add Missing Controls

**File**: `ui/desktop/src/controlwidget.h`

Add to private members section (after line 38):
```cpp
QLineEdit *symbolEdit;  // FIX: symbol input field
QSpinBox *batchSizeSpin;  // FIX: batch size spinner
QSpinBox *cvFoldsSpin;  // FIX: CV folds spinner
```

**File**: `ui/desktop/src/controlwidget.cpp`

In `setupUi()` method, after line 56 (after cvCheckbox), add:
```cpp
// FIX: Symbol input field per spec
symbolEdit = new QLineEdit();  // FIX: create line edit
symbolEdit->setText("BTCUSDT");  // FIX: default symbol
symbolEdit->setPlaceholderText("Trading pair (e.g., BTCUSDT)");  // FIX: placeholder
formLayout->addRow("Symbol:", symbolEdit);  // FIX: add to form

// FIX: Batch size spinner per spec
batchSizeSpin = new QSpinBox();  // FIX: create spin box
batchSizeSpin->setRange(1, 2048);  // FIX: valid batch size range
batchSizeSpin->setValue(512);  // FIX: default batch size
formLayout->addRow("Batch Size:", batchSizeSpin);  // FIX: add to form

// FIX: CV folds spinner per spec
cvFoldsSpin = new QSpinBox();  // FIX: create spin box
cvFoldsSpin->setRange(1, 20);  // FIX: CV folds range
cvFoldsSpin->setValue(5);  // FIX: default 5 folds
cvFoldsSpin->setEnabled(true);  // FIX: enable by default
formLayout->addRow("CV Folds:", cvFoldsSpin);  // FIX: add to form

// FIX: Connect CV checkbox to enable/disable folds spinner
connect(cvCheckbox, &QCheckBox::toggled, [this](bool checked) {
    cvFoldsSpin->setEnabled(checked);  // FIX: enable folds only when CV is checked
});
```

Add getter methods to retrieve values:
```cpp
// FIX: Add public getter methods to controlwidget.h
QString getSymbol() const { return symbolEdit->text(); }
int getBatchSize() const { return batchSizeSpin->value(); }
int getCVFolds() const { return cvFoldsSpin->value(); }
QString getTimeframe() const { return timeframeCombo->currentText(); }
int getHorizon() const { return horizonSpin->value(); }
QString getCalibrationMethod() const { return calibrationMethodCombo->currentText(); }
QString getPUpSource() const { return pUpSourceCombo->currentText(); }
bool isCVEnabled() const { return cvCheckbox->isChecked(); }
```

---

## 🔴 CRITICAL: REST + WebSocket Handshake

### Current Implementation (`ui/desktop/src/mainwindow.cpp`)

**Has:**
- ✅ WebSocket client connected to `ws://localhost:8000/ws`
- ✅ Signal `trainingStartRequested()` emitted when button clicked
- ❌ **NO REST POST call implemented**
- ❌ **NO WebSocket event parsing**
- ❌ **NO ACK timeout logic**

### Implementation Steps

#### Step 1: Add HTTP Client to MainWindow

**File**: `ui/desktop/src/mainwindow.h`

Add includes:
```cpp
#include <QNetworkAccessManager>  // FIX: HTTP client
#include <QNetworkRequest>  // FIX: HTTP request
#include <QNetworkReply>  // FIX: HTTP response
#include <QTimer>  // FIX: timeout timer
#include <QJsonDocument>  // FIX: JSON parsing
```

Add private members:
```cpp
QNetworkAccessManager *httpClient;  // FIX: HTTP client for REST calls
QTimer *ackTimeout;  // FIX: 30-second WebSocket ACK timer
QString currentJobId;  // FIX: track current training job_id
int restRetryCount;  // FIX: retry counter (max 2)
```

#### Step 2: Implement POST /start_training

**File**: `ui/desktop/src/mainwindow.cpp`

In constructor, initialize HTTP client:
```cpp
httpClient = new QNetworkAccessManager(this);  // FIX: create HTTP client
ackTimeout = new QTimer(this);  // FIX: create ACK timeout timer
ackTimeout->setSingleShot(true);  // FIX: fire once
connect(ackTimeout, &QTimer::timeout, this, &MainWindow::onAckTimeout);  // FIX: connect timeout

restRetryCount = 0;  // FIX: initialize retry counter
```

In `connectSignals()` method, update training start handler:
```cpp
connect(controlWidget, &ControlWidget::trainingStartRequested, this, [this]() {
    // FIX: Collect parameters from control panel
    QJsonObject requestBody;
    requestBody["symbol"] = controlWidget->getSymbol();
    requestBody["timeframe"] = controlWidget->getTimeframe();
    requestBody["batch_size"] = controlWidget->getBatchSize();
    requestBody["use_cv"] = controlWidget->isCVEnabled();
    requestBody["cv_folds"] = controlWidget->getCVFolds();
    requestBody["forecast_horizon"] = controlWidget->getHorizon();
    requestBody["calibration_method"] = controlWidget->getCalibrationMethod();
    requestBody["p_up_source"] = controlWidget->getPUpSource();
    requestBody["seed"] = 42;  // FIX: hardcoded seed for now

    // FIX: Client-side validation
    if (controlWidget->getSymbol().isEmpty()) {
        QMessageBox::warning(this, "Validation Error", "Symbol cannot be empty");
        return;
    }
    if (controlWidget->getBatchSize() <= 0) {
        QMessageBox::warning(this, "Validation Error", "Batch size must be positive");
        return;
    }

    // FIX: Disable start button, enable cancel button
    statusLabel->setText("Requesting training start...");

    // FIX: Send POST request
    QNetworkRequest request(QUrl("http://localhost:8000/start_training"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply *reply = httpClient->post(request, QJsonDocument(requestBody).toJson());

    // FIX: Handle REST response
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            statusLabel->setText(QString("Error: %1").arg(reply->errorString()));
            QMessageBox::critical(this, "Training Start Failed", reply->errorString());
            return;
        }

        // FIX: Parse response
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        QJsonObject response = doc.object();

        if (response["status"].toString() == "accepted") {
            currentJobId = response["job_id"].toString();  // FIX: save job_id
            statusLabel->setText("Training accepted — awaiting backend confirmation...");

            // FIX: Start 30-second ACK timeout
            ackTimeout->start(30000);  // FIX: 30 seconds
            restRetryCount = 0;  // FIX: reset retry counter
        } else {
            statusLabel->setText("Unexpected response from backend");
        }
    });
});
```

#### Step 3: Parse WebSocket Messages

In `connectSignals()`, add WebSocket message handler:
```cpp
connect(wsClient, &WebSocketClient::messageReceived, this, [this](const QString &message) {
    // FIX: Parse JSON message
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (!doc.isObject()) return;

    QJsonObject event = doc.object();
    QString eventType = event["type"].toString();

    // FIX: Handle training.started event
    if (eventType == "training.started" || eventType == "training_start") {
        QString jobId = event["job_id"].toString();

        // FIX: Verify job_id matches current request
        if (jobId == currentJobId || currentJobId.isEmpty()) {
            // FIX: Stop ACK timeout timer
            ackTimeout->stop();

            // FIX: Update GUI state
            statusLabel->setText("Training running");
            connectionLabel->setText("✓ Connected | Training active");

            // FIX: Enable stop button, disable start button
            // (controlWidget manages button states)

            qInfo() << "Training started confirmed via WebSocket";
        }
    }
    // FIX: Handle training.failed event
    else if (eventType == "training.failed" || eventType == "training_error") {
        ackTimeout->stop();
        QString reason = event["reason"].toString();
        statusLabel->setText(QString("Training failed: %1").arg(reason));
        QMessageBox::critical(this, "Training Failed", reason);
    }
    // FIX: Handle epoch metrics updates (existing logic)
    // ...
});
```

#### Step 4: Implement ACK Timeout with Retry

Add new slot to `mainwindow.cpp`:
```cpp
void MainWindow::onAckTimeout() {
    // FIX: No WebSocket ACK received within 30 seconds
    if (restRetryCount < 2) {
        // FIX: Retry REST call
        restRetryCount++;
        statusLabel->setText(QString("Retrying training start (attempt %1/3)...").arg(restRetryCount + 1));

        // FIX: Re-trigger training start (reuse same logic)
        qWarning() << "No WebSocket ACK, retrying..." << restRetryCount;

        // TODO: Retry logic here (call start_training again)

    } else {
        // FIX: Max retries exceeded, show error
        statusLabel->setText("Training start failed: no backend acknowledgement");

        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setWindowTitle("Training Start Failed");
        msgBox.setText("Unable to start training: backend did not acknowledge.");
        msgBox.setInformativeText("Check backend logs for details.");

        // FIX: Add "View Logs" button
        QPushButton *viewLogsBtn = msgBox.addButton("View Logs", QMessageBox::ActionRole);
        QPushButton *retryBtn = msgBox.addButton("Retry", QMessageBox::ActionRole);
        msgBox.addButton(QMessageBox::Cancel);

        msgBox.exec();

        if (msgBox.clickedButton() == viewLogsBtn) {
            // FIX: Call GET /status/log_tail and show in dialog
            showLogTailDialog();
        } else if (msgBox.clickedButton() == retryBtn) {
            // FIX: Reset and retry
            restRetryCount = 0;
            // TODO: Re-trigger training start
        }
    }
}
```

#### Step 5: Implement "View Logs" Dialog

```cpp
void MainWindow::showLogTailDialog() {
    // FIX: Fetch logs from backend
    QNetworkRequest request(QUrl("http://localhost:8000/status/log_tail?n=200"));
    QNetworkReply *reply = httpClient->get(request);

    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            QMessageBox::warning(this, "Log Fetch Failed", reply->errorString());
            return;
        }

        // FIX: Parse log lines
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        QJsonObject response = doc.object();
        QJsonArray lines = response["lines"].toArray();

        // FIX: Build log text
        QString logText;
        for (const QJsonValue &line : lines) {
            logText += line.toString() + "\n";
        }

        // FIX: Show in dialog
        QDialog *logDialog = new QDialog(this);
        logDialog->setWindowTitle("Backend Logs (Last 200 lines)");
        logDialog->resize(800, 600);

        QTextEdit *textEdit = new QTextEdit(logDialog);
        textEdit->setReadOnly(true);
        textEdit->setPlainText(logText);
        textEdit->setFont(QFont("Courier", 9));

        QVBoxLayout *layout = new QVBoxLayout(logDialog);
        layout->addWidget(textEdit);

        QPushButton *closeBtn = new QPushButton("Close", logDialog);
        connect(closeBtn, &QPushButton::clicked, logDialog, &QDialog::accept);
        layout->addWidget(closeBtn);

        logDialog->exec();
    });
}
```

---

## 🟡 Optional: Connection Health Indicators

Add to status bar:
- **REST Dot**: Poll `GET /` every 10 seconds
- **WebSocket Dot**: Monitor `wsClient->isConnected()`

Update colors:
- 🟢 Green = Connected/OK
- 🟡 Yellow = Reconnecting
- 🔴 Red = Disconnected/Error

---

## 📋 Build & Test

After making GUI changes:

```bash
# Rebuild GUI
cd ui/desktop/build
cmake ..
make -j4

# Test backend first
./scripts/debug_connection.sh

# Start backend
./start_backend.sh

# Start GUI
./start_gui.sh
```

---

## 🐛 Troubleshooting

1. **GUI doesn't compile**: Check Qt6 includes (QNetworkAccessManager, QNetworkRequest)
2. **Training doesn't start**: Check backend logs with `curl http://localhost:8000/status/log_tail`
3. **WebSocket not connecting**: Run `./scripts/debug_connection.sh`
4. **No ACK received**: Backend may need to emit event from training worker (not just REST endpoint)

---

## ✅ Acceptance Criteria

When complete:
- [ ] User clicks "Start Training" → GUI shows "Training accepted"
- [ ] Within 30s, GUI receives `training.started` event → changes to "Training running"
- [ ] If no event after 30s, GUI retries up to 2x, then shows error with "View Logs"
- [ ] Symbol, batch_size, cv_folds fields visible and functional
- [ ] Connection health indicators show REST + WebSocket status
- [ ] "View Logs" button fetches and displays backend logs

