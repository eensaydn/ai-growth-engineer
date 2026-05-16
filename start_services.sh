#!/usr/bin/env bash
# Start the API + cloudflared tunnel.
# Logs go to ./logs/. URL is printed to stdout once cloudflared finds one.
#
# Usage:  ./start_services.sh
# Stop:   ./stop_services.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs

if lsof -ti:8000 > /dev/null 2>&1; then
    echo "Port 8000 already in use — kill existing process first or run ./stop_services.sh"
    exit 1
fi

# 1) Start API
source venv/bin/activate
nohup python api.py > logs/api.log 2>&1 &
API_PID=$!
echo $API_PID > logs/api.pid
echo "API started (pid $API_PID), waiting for /health..."

for i in {1..15}; do
    if curl -s -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "API ready."
        break
    fi
    sleep 1
done

# 2) Start cloudflared
nohup cloudflared tunnel --url http://localhost:8000 > logs/cloudflared.log 2>&1 &
TUNNEL_PID=$!
echo $TUNNEL_PID > logs/cloudflared.pid
echo "cloudflared started (pid $TUNNEL_PID), waiting for public URL..."

PUBLIC_URL=""
for i in {1..30}; do
    PUBLIC_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/cloudflared.log 2>/dev/null | head -1 || true)
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$PUBLIC_URL" ]; then
    echo "WARNING: Could not detect public URL in logs/cloudflared.log within 30s."
    echo "Check the log:  tail -f logs/cloudflared.log"
    exit 1
fi

echo "$PUBLIC_URL" > logs/public_url.txt

cat <<EOF

────────────────────────────────────────────────────────────
  ✓ Services running
────────────────────────────────────────────────────────────
  Local API     : http://127.0.0.1:8000
  Swagger UI    : http://127.0.0.1:8000/docs
  Public URL    : $PUBLIC_URL
  Retool base   : $PUBLIC_URL     (use this in Retool Resource)
  Health check  : $PUBLIC_URL/health
────────────────────────────────────────────────────────────
  Logs          : logs/api.log, logs/cloudflared.log
  Stop          : ./stop_services.sh
────────────────────────────────────────────────────────────
EOF
