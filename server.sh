#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"

load_env() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo "Error: .env file not found. Copy .env.example to .env and configure."
        exit 1
    fi
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server is already running (PID: $(cat "$PID_FILE"))"
        return
    fi

    echo "Starting server..."
    load_env

    cd "$SCRIPT_DIR"
    nohup python3 backend.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2

    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server started (PID: $(cat "$PID_FILE"), Port: ${PORT:-8080})"
    else
        echo "Server failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        PID=$(lsof -ti :${PORT:-8080} 2>/dev/null || true)
        if [ -n "$PID" ]; then
            echo "Stopping server (PID: $PID)..."
            kill "$PID"
            sleep 1
            echo "Server stopped"
        else
            echo "Server is not running"
        fi
        return
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        sleep 1
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        echo "Server stopped"
    else
        echo "Server is not running (stale PID file)"
    fi
    rm -f "$PID_FILE"
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Server is running (PID: $PID, Port: ${PORT:-8080})"
        else
            echo "Server is not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        PID=$(lsof -ti :${PORT:-8080} 2>/dev/null || true)
        if [ -n "$PID" ]; then
            echo "Server is running (PID: $PID, Port: ${PORT:-8080})"
        else
            echo "Server is stopped"
        fi
    fi
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac