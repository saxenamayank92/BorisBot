# BorisBot browser image scaffold for remote desktop and CDP access.
FROM mcr.microsoft.com/playwright/python:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080
ENV CDP_PORT=9222

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu \
        fluxbox \
        novnc \
        pcmanfm \
        websockify \
        x11vnc \
        xvfb \
    && mkdir -p /browser-profile \
    && rm -rf /var/lib/apt/lists/*

COPY vnc-start.sh /usr/local/bin/vnc-start.sh
COPY fluxbox-config/init /etc/fluxbox/init

RUN chmod +x /usr/local/bin/vnc-start.sh

EXPOSE 9222 5900 6080

CMD ["/usr/local/bin/vnc-start.sh"]
