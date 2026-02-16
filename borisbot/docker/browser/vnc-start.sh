#!/usr/bin/env bash
# Start Xvfb, fluxbox, Chromium and x11vnc for browser container scaffolding.
set -euo pipefail

log() {
  printf '[browser-image] %s\n' "$*"
}

log "Starting Xvfb on ${DISPLAY}"
Xvfb "${DISPLAY}" -screen 0 1280x800x24 &

log "Starting fluxbox"
fluxbox -display "${DISPLAY}" &

log "Starting Chromium with CDP on port ${CDP_PORT}"
chromium-browser \
  --no-sandbox \
  --disable-dev-shm-usage \
  --remote-debugging-address=0.0.0.0 \
  --remote-debugging-port="${CDP_PORT}" \
  --user-data-dir=/tmp/chromium-profile \
  --disable-gpu \
  --no-first-run \
  --no-default-browser-check \
  --disable-background-networking \
  about:blank >/tmp/chromium.log 2>&1 &

log "Starting x11vnc on port ${VNC_PORT}"
exec x11vnc \
  -display "${DISPLAY}" \
  -forever \
  -shared \
  -rfbport "${VNC_PORT}" \
  -nopw
