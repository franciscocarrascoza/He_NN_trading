# WebSocket Message Schema

This document describes the JSON message schema for real-time training updates via the WebSocket endpoint `/ws`.

---

## Connection

**Endpoint**: `ws://localhost:8000/ws` (default)

**Protocol**: WebSocket (RFC 6455)

**Format**: JSON text messages

**Client Requirements**:
- Send periodic ping messages to keep connection alive
- Handle reconnection on disconnect

---

## Message Types

All messages are JSON objects with a `type` field identifying the message type.

### 1. `dataset_ready`

Sent when the dataset has been prepared and training is ready to start.

```json
{
  "type": "dataset_ready",
  "size": 10000,
  "min_samples": 2048
}
```

**Fields**:
- `type` (string): Always `"dataset_ready"`
- `size` (int): Total number of samples in dataset
- `min_samples` (int): Minimum samples required for training

---

### 2. `training_start`

Sent when training begins.

```json
{
  "type": "training_start",
  "use_cv": true,
  "total_folds": 5,
  "total_epochs": 25
}
```

**Fields**:
- `type` (string): Always `"training_start"`
- `use_cv` (bool): Whether cross-validation is enabled
- `total_folds` (int): Total number of CV folds (1 if CV disabled)
- `total_epochs` (int): Total epochs per fold

---

### 3. `epoch_metrics`

Sent at the end of each training epoch with validation metrics.

```json
{
  "type": "epoch_metrics",
  "fold_id": 0,
  "epoch": 10,
  "train_loss": 0.1234,
  "train_nll": 0.0987,
  "train_bce": 0.0234,
  "train_brier": 0.1345,
  "val_loss": 0.2345,
  "val_nll": 0.1987,
  "val_bce": 0.0456,
  "val_brier": 0.1567,
  "val_auc": 0.6789,
  "DirAcc": 0.5678,
  "ECE": 0.0456,
  "MZ_intercept": 0.0012,
  "MZ_slope": 0.9987,
  "MZ_F_p": 0.3456,
  "PIT_KS_p": 0.4567,
  "conformal_coverage": 0.9012,
  "conformal_width": 0.1234,
  "elapsed_seconds": 123.45
}
```

**Fields**:
- `type` (string): Always `"epoch_metrics"`
- `fold_id` (int): Current fold ID (0-indexed)
- `epoch` (int): Current epoch number (0-indexed)
- `train_loss` (float): Training loss (combined)
- `train_nll` (float): Training NLL component
- `train_bce` (float): Training BCE component
- `train_brier` (float): Training Brier score
- `val_loss` (float): Validation loss (combined)
- `val_nll` (float): Validation NLL component
- `val_bce` (float): Validation BCE component
- `val_brier` (float): Validation Brier score
- `val_auc` (float): Validation AUC-ROC
- `DirAcc` (float): Directional accuracy
- `ECE` (float): Expected Calibration Error
- `MZ_intercept` (float): Mincer-Zarnowitz regression intercept
- `MZ_slope` (float): Mincer-Zarnowitz regression slope
- `MZ_F_p` (float): Mincer-Zarnowitz F-test p-value
- `PIT_KS_p` (float): PIT z-score Kolmogorov-Smirnov test p-value
- `conformal_coverage` (float): Conformal interval empirical coverage
- `conformal_width` (float): Mean conformal interval width
- `elapsed_seconds` (float): Elapsed time since training start

---

### 4. `fold_complete`

Sent when a fold completes training and evaluation.

```json
{
  "type": "fold_complete",
  "fold_id": 0,
  "predictions_path": "reports/predictions/predictions_fold_0_20230101_120000.csv",
  "calibration_method": "temp_isotonic",
  "calibration_warning": null,
  "coverage_warning": null
}
```

**Fields**:
- `type` (string): Always `"fold_complete"`
- `fold_id` (int): Completed fold ID
- `predictions_path` (string): Path to predictions CSV
- `calibration_method` (string): Selected calibration method (`"raw"`, `"temperature"`, `"isotonic"`, `"temp_isotonic"`)
- `calibration_warning` (string|null): Calibration warning message if any
- `coverage_warning` (string|null): Conformal coverage warning if out of tolerance

---

### 5. `training_complete`

Sent when all training folds complete.

```json
{
  "type": "training_complete",
  "folds_completed": 5,
  "summary_path": "reports/summary.json"
}
```

**Fields**:
- `type` (string): Always `"training_complete"`
- `folds_completed` (int): Number of completed folds
- `summary_path` (string): Path to summary JSON report

---

### 6. `training_error`

Sent when training encounters an error and stops.

```json
{
  "type": "training_error",
  "error": "RuntimeError: CUDA out of memory"
}
```

**Fields**:
- `type` (string): Always `"training_error"`
- `error` (string): Error message

---

### 7. `prediction_delta` (Future Extension)

Future extension: incremental prediction updates during training.

```json
{
  "type": "prediction_delta",
  "fold_id": 0,
  "epoch": 10,
  "predictions": [
    {
      "timestamp": "2023-01-01T00:00:00Z",
      "mu": 0.0123,
      "sigma": 0.0234,
      "p_up_raw": 0.5234,
      "p_up_cal": 0.5123,
      "conformal_p": 0.8765,
      "pit_z": 0.1234
    }
  ]
}
```

**Fields**:
- `type` (string): Always `"prediction_delta"`
- `fold_id` (int): Fold ID
- `epoch` (int): Epoch number
- `predictions` (array): Array of prediction objects

**Note**: Not implemented in current version. Reserved for future use.

---

## Client Implementation Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('WebSocket connected');
  // Send periodic ping to keep connection alive
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 30000); // Ping every 30 seconds
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'training_start':
      console.log(`Training started: ${message.total_folds} folds, ${message.total_epochs} epochs`);
      break;

    case 'epoch_metrics':
      console.log(`Epoch ${message.epoch}: AUC=${message.val_auc}, DirAcc=${message.DirAcc}`);
      // Update UI with metrics
      break;

    case 'training_complete':
      console.log('Training complete');
      // Download summary report
      break;

    case 'training_error':
      console.error(`Training error: ${message.error}`);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
  // Implement reconnection logic
};
```

---

## Client Implementation Example (Qt C++)

```cpp
// FIX: Qt WebSocket client example from websocketclient.cpp

QWebSocket *webSocket = new QWebSocket();

connect(webSocket, &QWebSocket::textMessageReceived, [](const QString &message) {
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (doc.isNull() || !doc.isObject()) {
        qWarning() << "Invalid WebSocket message JSON";
        return;
    }

    QJsonObject obj = doc.object();
    QString messageType = obj["type"].toString();

    if (messageType == "epoch_metrics") {
        double auc = obj["val_auc"].toDouble();
        double dirAcc = obj["DirAcc"].toDouble();
        qDebug() << "Epoch metrics: AUC=" << auc << "DirAcc=" << dirAcc;
        // Update metrics panel
    }
    else if (messageType == "training_complete") {
        qDebug() << "Training complete";
        // Update UI state
    }
});

webSocket->open(QUrl("ws://localhost:8000/ws"));
```

---

## Error Handling

### Connection Errors

If the WebSocket connection fails:
1. Verify backend is running: `curl http://localhost:8000/`
2. Check firewall/network settings
3. Implement exponential backoff for reconnection

### Message Parse Errors

If JSON parsing fails:
1. Log the raw message for debugging
2. Continue processing subsequent messages
3. Report parsing error to backend logs

### Backend Errors

If backend sends `training_error`:
1. Display error to user
2. Allow user to restart training with adjusted parameters
3. Check backend logs for detailed stack trace

---

## Rate Limiting

The WebSocket endpoint has no explicit rate limiting, but:
- Backend sends messages only during training (epoch boundaries)
- Typical message rate: 1-10 messages per minute
- Client should handle bursts gracefully (e.g., at epoch end)

---

## Security Considerations

**Current Implementation** (local development):
- No authentication required
- No encryption (ws://)
- Suitable for localhost-only deployment

**Production Deployment** (future):
- Add authentication via JWT or session tokens
- Use WSS (WebSocket Secure) with TLS
- Implement rate limiting and connection limits

---

## Testing WebSocket Connection

### Using `wscat` (Node.js tool)

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket endpoint
wscat -c ws://localhost:8000/ws

# Send ping message
> ping

# Observe incoming messages during training
```

### Using Python `websockets` library

```python
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected")

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received {data['type']}: {data}")

asyncio.run(test_ws())
```

---

## Future Extensions

Planned WebSocket message extensions:

1. **Per-Batch Progress** (`batch_progress`)
   - Real-time progress within epoch
   - Useful for long epochs

2. **Hyperparameter Trial Updates** (`hpo_trial`)
   - Optuna trial suggestions and results
   - HPO study progress

3. **Live Chart Data** (`chart_update`)
   - Incremental OHLCV candle updates
   - Reduces need for full chart reload

4. **User Notifications** (`notification`)
   - Training milestones (50% complete, best AUC achieved)
   - Warnings and alerts

---

## References

- **WebSocket RFC 6455**: [https://tools.ietf.org/html/rfc6455](https://tools.ietf.org/html/rfc6455)
- **FastAPI WebSockets Guide**: [https://fastapi.tiangolo.com/advanced/websockets/](https://fastapi.tiangolo.com/advanced/websockets/)
- **Qt WebSockets Documentation**: [https://doc.qt.io/qt-6/qtwebsockets-index.html](https://doc.qt.io/qt-6/qtwebsockets-index.html)

---

**Last Updated**: 2025-01-23
