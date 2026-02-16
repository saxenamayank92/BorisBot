# BorisBot browser image scaffold for remote desktop and Chrome CDP access.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV VNC_PORT=5901
ENV CDP_PORT=9222

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium-browser \
        dbus-x11 \
        fluxbox \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY vnc-start.sh /usr/local/bin/vnc-start.sh
COPY fluxbox-config/init /etc/fluxbox/init

RUN chmod +x /usr/local/bin/vnc-start.sh

EXPOSE 5901 9222

CMD ["/usr/local/bin/vnc-start.sh"]
