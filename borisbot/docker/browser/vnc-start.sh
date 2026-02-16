#!/usr/bin/env bash
# Start Xvfb, fluxbox, browser CDP, VNC and noVNC for container scaffolding.
set -euo pipefail

log() {
  printf '[browser-image] %s\n' "$*"
}

export DISPLAY=:99
mkdir -p /browser-profile

CHROME_BIN="$(ls -d /ms-playwright/chromium-*/chrome-linux/chrome 2>/dev/null | head -n 1)"
if [[ -z "${CHROME_BIN}" ]]; then
  log "Playwright Chromium binary not found under /ms-playwright; exiting."
  exit 1
fi

log "Starting Xvfb on ${DISPLAY}"
Xvfb "${DISPLAY}" -screen 0 1280x800x24 &

log "Starting fluxbox"
fluxbox -display "${DISPLAY}" &

log "Starting Playwright Chromium with CDP on port 9222"
"${CHROME_BIN}" \
  --remote-debugging-port=9222 \
  --remote-debugging-address=0.0.0.0 \
  --no-sandbox \
  --disable-dev-shm-usage \
  --user-data-dir=/browser-profile \
  about:blank >/tmp/chromium.log 2>&1 &

log "Starting x11vnc on port ${VNC_PORT}"
x11vnc \
  -display "${DISPLAY}" \
  -forever \
  -shared \
  -rfbport "${VNC_PORT}" \
  -nopw >/tmp/x11vnc.log 2>&1 &

log "Starting noVNC via websockify on port 6080"
websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &

wait -n
