# Backend Fixes Applied - Training Handshake & Diagnostics

## ✅ Completed Backend Fixes (2025-01-24)

### 1. **Training Start Endpoint Enhanced** (`/start_training`)
- **Returns HTTP 202 Accepted** (not 200) per spec
- **Generates unique `job_id`** for each training run
- **Emits `training.started` WebSocket event** immediately after accepting request
- **Added parameters**: `symbol`, `timeframe`, `batch_size` to request schema
- **Response includes**: `job_id`, `expected_pid`, status message

**Event Format:**
```json
{
  "type": "training.started",
  "job_id": "uuid-here",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "epoch": 0,
  "fold": 0,
  "timestamp": 1234567890.123
}
```

### 2. **New Diagnostic Endpoint** (`/status/log_tail`)
- **GET `/status/log_tail?n=200`** - Returns last N lines of backend log
- **Response format**:
```json
{
  "lines": ["log line 1", "log line 2", ...],
  "count": 150,
  "requested": 200
}
```
- Enables "View Logs" button in GUI for troubleshooting
- Falls back gracefully if `backend.log` doesn't exist

### 3. **WebSocket Event Emission**
- Backend now broadcasts `training.started` to all connected WebSocket clients
- Event sent immediately after REST `/start_training` returns 202
- GUI can listen for this event to confirm training actually started
- Handles WebSocket send failures gracefully with warnings

---

## 🔧 Testing the Fixes

### Test REST Endpoint:
```bash
curl -X POST http://localhost:8000/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "batch_size": 512,
    "use_cv": true,
    "cv_folds": 5,
    "forecast_horizon": 1,
    "calibration_method": "abs",
    "p_up_source": "cdf",
    "seed": 42
  }'
```

**Expected Response:**
```json
{
  "status": "accepted",
  "job_id": "a1b2c3d4-...",
  "message": "Training worker queued and starting",
  "expected_pid": 12345
}
```

### Test Log Tail Endpoint:
```bash
curl http://localhost:8000/status/log_tail?n=50
```

### Test Connection Health:
```bash
./scripts/debug_connection.sh
```

---

## 🎯 What GUI Needs to Implement

### 1. **Enhanced Control Panel** (Partially Done)
Current controls work but need additions:
- ✅ Timeframe dropdown
- ✅ Forecast horizon spinner
- ✅ Calibration method dropdown
- ✅ p_up source dropdown
- ✅ CV checkbox
- ❌ **MISSING**: Symbol input field (default: "BTCUSDT")
- ❌ **MISSING**: Batch size spinner (default: 512)
- ❌ **MISSING**: CV folds spinner (1-20, default: 5)

### 2. **REST + WebSocket Handshake Flow**
When user clicks "Start Training":

```
[User clicks Start]
    ↓
[GUI: POST /start_training with all parameters]
    ↓
[Backend: Returns 202 Accepted with job_id]
    ↓
[GUI: Shows "Training accepted - awaiting confirmation..."]
[GUI: Starts 30-second WebSocket ACK timer]
    ↓
[Backend: Emits training.started via WebSocket]
    ↓
[GUI: Receives WS event, validates job_id matches]
    ↓
[GUI: Changes state to "Training Running"]
[GUI: Enables "Stop Training" button]
[GUI: Starts listening for epoch/metrics updates]
```

**Timeout Handling:**
- If no `training.started` event within 30 seconds:
  - Retry REST call up to 2 times
  - If still no ACK, show error dialog:
    ```
    "Unable to start training: backend did not acknowledge.
    Check backend logs for details."
    [View Logs] [Retry] [Cancel]
    ```

### 3. **Connection Health Indicators**
Add status indicators to GUI status bar:
- **REST Health**: Green dot = reachable, Red = unreachable
- **WebSocket Health**: Green = connected, Yellow = reconnecting, Red = disconnected
- Poll REST `/` endpoint every 10 seconds
- Show last successful ping time

### 4. **"View Logs" Button**
When user clicks "View Logs":
- Call `GET /status/log_tail?n=200`
- Display in scrollable text area or separate dialog
- Auto-refresh every 5 seconds while dialog is open

---

## 📋 Implementation Checklist

### Backend (Completed)
- [x] Add `symbol`, `timeframe`, `batch_size` to TrainingStartRequest
- [x] Return HTTP 202 with `job_id` from `/start_training`
- [x] Emit `training.started` WebSocket event
- [x] Implement `/status/log_tail` endpoint
- [x] Add `training.failed` event emission (TODO: add to worker on errors)

### GUI (Remaining)
- [ ] Add missing control panel fields (symbol, batch_size, cv_folds)
- [ ] Implement REST POST `/start_training` on button click
- [ ] Parse 202 response and extract `job_id`
- [ ] Add WebSocket message parser for `training.started` event
- [ ] Implement 30-second ACK timeout with retry logic
- [ ] Add connection health indicators (REST/WS status dots)
- [ ] Implement "View Logs" button → calls `/status/log_tail`
- [ ] Add "Cancel start request" button during pending state

### Scripts (Completed)
- [x] Create `scripts/debug_connection.sh` for diagnostics
- [ ] Create `scripts/migrate_old_config.py` for config migration

### Layout System (TODO)
- [ ] Add layout fallback system (corrupt JSON handling)
- [ ] Backup existing layout to `.bak.TIMESTAMP`
- [ ] Create `.default` file when corrupted
- [ ] Show non-blocking warning popup in GUI

---

## 🐛 Known Issues & Next Steps

1. **Training Worker**: May need to emit `training.started` again from worker.run()
   - Currently emitted from REST endpoint only
   - Worker might fail to start after REST returns

2. **Error Handling**: Need `training.failed` event emission
   - Add try/catch in worker.run()
   - Emit `training.failed` with error details

3. **Job Status Endpoint**: Not yet implemented
   - Add `GET /jobs/{job_id}` for polling job status
   - Useful as fallback if WebSocket fails

---

## 📝 Backend Code Changes Summary

**File**: `backend/app.py`
- Lines 54-65: Added `symbol`, `timeframe`, `batch_size` to `TrainingStartRequest`
- Lines 96-174: Enhanced `/start_training` endpoint:
  - Generates `job_id`
  - Returns HTTP 202 instead of 200
  - Broadcasts `training.started` WebSocket event
- Lines 252-281: New `/status/log_tail` endpoint for diagnostics

**Total Lines Changed**: ~50 lines added/modified in backend

---

## 🚀 Quick Start After These Fixes

```bash
# Terminal 1: Start backend
./start_backend.sh

# Terminal 2: Test connection
./scripts/debug_connection.sh

# Terminal 3: Start GUI (needs GUI updates first)
./start_gui.sh
```

---

## 📞 Support

If training still doesn't start after GUI updates:
1. Run `./scripts/debug_connection.sh`
2. Check `curl http://localhost:8000/status/log_tail`
3. Monitor WebSocket in browser console or with `wscat -c ws://localhost:8000/ws`
4. Check for `training.started` event emission in backend logs

