#!/bin/bash
# Quick check on the perf run status
# Usage: bash scripts/check_perf_status.sh

LOG="/home/azureuser/clustering-v2/hackathon/data/public_contracts_perf.log"
REPORT="/home/azureuser/clustering-v2/hackathon/data/public_contracts_perf_report.txt"

echo "=== PERF RUN STATUS ==="
echo ""

# Check if report exists (means run completed)
if [ -f "$REPORT" ]; then
    echo "✅ RUN COMPLETED — Full report:"
    echo ""
    cat "$REPORT"
    exit 0
fi

# Check if process is still running
if pgrep -f "run_public_contracts_perf" > /dev/null; then
    echo "⏳ STILL RUNNING"
    echo ""
    
    # Show latest progress
    if [ -f "$LOG" ]; then
        echo "--- Latest log lines ---"
        tail -15 "$LOG"
        echo ""
        echo "--- Progress counts ---"
        echo "GATE PASSED: $(grep -c 'GATE PASSED' "$LOG" 2>/dev/null || echo 0)"
        echo "GATE FAILED: $(grep -c 'GATE FAILED' "$LOG" 2>/dev/null || echo 0)"
        echo "Intents persisted: $(grep -c 'intents persisted' "$LOG" 2>/dev/null || echo 0)"
        echo "RAG extractions: $(grep -c 'extractions (RAG' "$LOG" 2>/dev/null || echo 0)"
        echo ""
        echo "Last cluster:"
        grep 'discovery\]   \[' "$LOG" 2>/dev/null | tail -1
    fi
else
    echo "❌ NOT RUNNING (may have crashed)"
    if [ -f "$LOG" ]; then
        echo ""
        echo "--- Last 20 log lines ---"
        tail -20 "$LOG"
    fi
fi
