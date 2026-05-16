#!/usr/bin/env bash
# Stop the API + cloudflared services started by ./start_services.sh

cd "$(dirname "$0")"

for name in api cloudflared; do
    pid_file=logs/${name}.pid
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid && echo "Stopped $name (pid $pid)."
        else
            echo "$name (pid $pid) was not running."
        fi
        rm -f "$pid_file"
    fi
done

# Fallback for any process bound to port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill 2>/dev/null && echo "Killed leftover process on :8000."
fi

rm -f logs/public_url.txt
echo "All stopped."
