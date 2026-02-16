#!/usr/bin/env bash
# Start Xvfb, fluxbox, browser CDP, VNC and noVNC for container scaffolding.
set -euo pipefail

log() {
  printf '[browser-image] %s\n' "$*"
}

export DISPLAY=:99
mkdir -p /browser-profile

find_browser_bin() {
  if command -v chromium >/dev/null 2>&1; then
    printf 'chromium'
    return 0
  fi
  if command -v chromium-browser >/dev/null 2>&1; then
    printf 'chromium-browser'
    return 0
  fi
  if command -v google-chrome >/dev/null 2>&1; then
    printf 'google-chrome'
    return 0
  fi
  return 1
}

log "Starting Xvfb on ${DISPLAY}"
Xvfb "${DISPLAY}" -screen 0 1280x800x24 &

log "Starting fluxbox"
fluxbox -display "${DISPLAY}" &

if BROWSER_BIN="$(find_browser_bin)"; then
  log "Starting ${BROWSER_BIN} with CDP on port ${CDP_PORT}"
  "${BROWSER_BIN}" \
    --no-sandbox \
    --disable-dev-shm-usage \
    --remote-debugging-address=0.0.0.0 \
    --remote-debugging-port="${CDP_PORT}" \
    --user-data-dir=/browser-profile \
    --disable-gpu \
    --no-first-run \
    --no-default-browser-check \
    --disable-background-networking \
    about:blank >/tmp/chromium.log 2>&1 &
else
  log "No Chromium-compatible binary found on PATH; skipping CDP process startup."
fi

log "Starting x11vnc on port ${VNC_PORT}"
x11vnc \
  -display "${DISPLAY}" \
  -forever \
  -shared \
  -rfbport "${VNC_PORT}" \
  -nopw >/tmp/x11vnc.log 2>&1 &

NOVNC_PROXY="/usr/share/novnc/utils/novnc_proxy"
if [[ ! -x "${NOVNC_PROXY}" ]]; then
  log "noVNC proxy not found at ${NOVNC_PROXY}; exiting."
  exit 1
fi
log "Starting noVNC proxy on port ${NOVNC_PORT}"
"${NOVNC_PROXY}" --vnc "localhost:${VNC_PORT}" --listen "${NOVNC_PORT}" >/tmp/novnc.log 2>&1 &

wait -n
