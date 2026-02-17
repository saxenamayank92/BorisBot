(() => {
  "use strict";

  try {
    // Record only top-level document interactions; ignore iframes.
    try {
      if (window.top !== window.self) return;
    } catch (_) {
      return;
    }

    const RECORD_HOST = window.__BORIS_RECORD_HOST__ || "host.docker.internal";
    const RECORD_PORT = window.__BORIS_RECORD_PORT__ || 7331;
    const ENDPOINT = `http://${RECORD_HOST}:${RECORD_PORT}/event`;
    let lastNavigatedUrl = null;
    const pendingInputValues = new Map();

    const isStableId = (id) => {
      if (!id || typeof id !== "string") return false;
      if (id.length > 64) return false;
      const lowered = id.toLowerCase();
      if (
        lowered.startsWith("react") ||
        lowered.startsWith("headlessui") ||
        lowered.includes(":r") ||
        lowered.includes("chakra")
      ) {
        return false;
      }
      return /^[a-zA-Z][a-zA-Z0-9_-]*$/.test(id);
    };

    const firstStableClass = (el) => {
      if (!el || !el.classList) return null;
      for (const c of el.classList) {
        if (!c) continue;
        if (c.length > 48) continue;
        if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(c)) continue;
        if (/[0-9]{3,}/.test(c)) continue;
        if (c.includes("css-") || c.includes("sc-")) continue;
        return c;
      }
      return null;
    };

    const toSelector = (element) => {
      if (!element || !element.tagName) return "body";

      const attr = (name) => element.getAttribute(name);
      const tag = element.tagName.toLowerCase();

      const dataTestId = attr("data-testid");
      if (dataTestId) return `[data-testid="${CSS.escape(dataTestId)}"]`;

      const dataTest = attr("data-test");
      if (dataTest) return `[data-test="${CSS.escape(dataTest)}"]`;

      const dataQa = attr("data-qa");
      if (dataQa) return `[data-qa="${CSS.escape(dataQa)}"]`;

      if (isStableId(element.id)) return `#${CSS.escape(element.id)}`;

      const ariaLabel = attr("aria-label");
      if (ariaLabel) return `[aria-label="${CSS.escape(ariaLabel)}"]`;

      const name = attr("name");
      if (name) return `[name="${CSS.escape(name)}"]`;

      const role = attr("role");
      if (role) return `[role="${CSS.escape(role)}"]`;

      const cls = firstStableClass(element);
      if (cls) return `${tag}.${CSS.escape(cls)}`;

      return tag;
    };

    const sendEvent = (eventType, payload) => {
      try {
        const event = {
          event_type: eventType,
          payload: payload || {},
          created_at: new Date().toISOString(),
        };
        if (typeof window.__BORIS_RECORD_EVENT__ === "function") {
          try {
            window.__BORIS_RECORD_EVENT__(event);
          } catch (_) {
            // fallback to HTTP transport
          }
        }
        const body = JSON.stringify(event);
        fetch(ENDPOINT, {
          method: "POST",
          mode: "no-cors",
          cache: "no-store",
          keepalive: true,
          headers: { "Content-Type": "text/plain;charset=UTF-8" },
          body,
        }).catch(() => {});
      } catch (_) {
        // never crash page runtime
      }
    };

    const emitNavigateIfChanged = () => {
      try {
        const url = window.location.href;
        if (!url || url === lastNavigatedUrl) return;
        lastNavigatedUrl = url;
        sendEvent("navigate", { url });
      } catch (_) {
        // never crash page runtime
      }
    };

    const installNavigationHooks = () => {
      const origPushState = history.pushState;
      const origReplaceState = history.replaceState;

      history.pushState = function (...args) {
        const result = origPushState.apply(this, args);
        emitNavigateIfChanged();
        return result;
      };

      history.replaceState = function (...args) {
        const result = origReplaceState.apply(this, args);
        emitNavigateIfChanged();
        return result;
      };

      window.addEventListener("popstate", emitNavigateIfChanged, true);
      window.addEventListener("hashchange", emitNavigateIfChanged, true);
      document.addEventListener("DOMContentLoaded", emitNavigateIfChanged, true);
      window.addEventListener("load", emitNavigateIfChanged, true);
    };

    document.addEventListener(
      "click",
      (event) => {
        try {
          const target = event.target && event.target.closest
            ? event.target.closest("a,button,[role],input,textarea,select,[data-testid],[data-test],[data-qa],*[aria-label],*[name]")
            : null;
          if (!target) return;
          const selector = toSelector(target);
          sendEvent("click", { selector });
        } catch (_) {
          // never crash page runtime
        }
      },
      true
    );

    document.addEventListener(
      "input",
      (event) => {
        try {
          const target = event.target;
          if (!target || !(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) {
            return;
          }
          const selector = toSelector(target);
          pendingInputValues.set(selector, target.value ?? "");
        } catch (_) {
          // never crash page runtime
        }
      },
      true
    );

    document.addEventListener(
      "blur",
      (event) => {
        try {
          const target = event.target;
          if (!target || !(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) {
            return;
          }
          const selector = toSelector(target);
          const value = pendingInputValues.has(selector)
            ? pendingInputValues.get(selector)
            : target.value ?? "";
          pendingInputValues.delete(selector);
          sendEvent("type", { selector, text: value });
        } catch (_) {
          // never crash page runtime
        }
      },
      true
    );

    installNavigationHooks();
    emitNavigateIfChanged();
  } catch (_) {
    // never crash page runtime
  }
})();
