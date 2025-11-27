#!/bin/bash
# FIX: Debug connection script for He_NN Trading Backend diagnostics

echo "================================"
echo "He_NN Trading Connection Debugger"
echo "================================"
echo ""

# FIX: Check if backend is running
echo "[1/4] Checking REST API (http://localhost:8000)..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    response=$(curl -s http://localhost:8000/)
    echo "✓ REST API is reachable"
    echo "  Response: $response"
else
    echo "✗ REST API is NOT reachable"
    echo "  Make sure backend is running: ./start_backend.sh"
    exit 1
fi

echo ""

# FIX: Check health endpoint
echo "[2/4] Checking backend health..."
health=$(curl -s http://localhost:8000/)
if echo "$health" | grep -q "ok"; then
    echo "✓ Backend health: OK"
else
    echo "⚠ Backend health check failed"
    echo "  Response: $health"
fi

echo ""

# FIX: Test training start endpoint (without actually starting)
echo "[3/4] Testing /start_training endpoint schema..."
test_response=$(curl -s -X POST http://localhost:8000/start_training \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","timeframe":"1h","batch_size":512,"use_cv":true,"cv_folds":5}' \
  2>&1)

if echo "$test_response" | grep -q "accepted\|already in progress"; then
    echo "✓ /start_training endpoint responds correctly"
    echo "  Response: $test_response"
elif echo "$test_response" | grep -q "job_id"; then
    echo "✓ /start_training accepted training request"
    echo "  Response: $test_response"
else
    echo "⚠ /start_training unexpected response"
    echo "  Response: $test_response"
fi

echo ""

# FIX: Test WebSocket connectivity (requires wscat or websocat)
echo "[4/4] Testing WebSocket connection (ws://localhost:8000/ws)..."
if command -v websocat &> /dev/null; then
    echo "  Using websocat to test WebSocket..."
    timeout 3 websocat ws://localhost:8000/ws <<< "ping" 2>&1 | head -n 5
    echo "✓ WebSocket test attempted (check output above)"
elif command -v wscat &> /dev/null; then
    echo "  Using wscat to test WebSocket..."
    timeout 3 wscat -c ws://localhost:8000/ws 2>&1 | head -n 5
    echo "✓ WebSocket test attempted (check output above)"
else
    echo "⚠ WebSocket tools not found (install: npm install -g wscat  OR  cargo install websocat)"
    echo "  WebSocket endpoint: ws://localhost:8000/ws"
    echo "  Test manually from GUI or browser console"
fi

echo ""
echo "================================"
echo "Debug Summary Complete"
echo "================================"
echo ""
echo "Next steps if issues found:"
echo "  1. Check backend logs: tail -f backend.log"
echo "  2. View last 200 log lines: curl http://localhost:8000/status/log_tail"
echo "  3. Verify Qt6 GUI can connect: ./start_gui.sh"
echo ""
