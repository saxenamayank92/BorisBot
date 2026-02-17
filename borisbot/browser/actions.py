"""Deterministic browser action primitives built on BrowserExecutor."""

import asyncio
import logging

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright._impl._errors import Error as PlaywrightError

from borisbot.browser.executor import BrowserExecutor

logger = logging.getLogger("borisbot.browser.actions")


class BrowserActions:
    """Thin deterministic action layer over BrowserExecutor."""

    def __init__(self, executor: BrowserExecutor):
        self.executor = executor

    async def safe_navigate(self, url: str, timeout: int = 15_000) -> None:
        """Navigate with DOM-first readiness and graceful networkidle fallback."""
        page = self.executor._require_page()

        await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)

        try:
            await page.wait_for_load_state("networkidle", timeout=3_000)
        except PlaywrightTimeoutError:
            pass

        stable_count = 0
        last_url = None
        for _ in range(10):
            current_url = page.url
            if current_url == last_url:
                stable_count += 1
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
                last_url = current_url
            await asyncio.sleep(0.2)

        title = (await page.title()).strip()
        if not title:
            raise RuntimeError("Page loaded with empty title")

    async def _execute_click(self, selector: str, timeout: int) -> None:
        """Execute one deterministic click attempt with stabilization checks."""
        page = self.executor._require_page()
        locator = page.locator(selector)

        await locator.wait_for(state="visible", timeout=timeout)
        await locator.wait_for(state="attached", timeout=timeout)
        await page.wait_for_function(
            "(sel) => { const el = document.querySelector(sel); return !!el && !el.disabled; }",
            arg=selector,
            timeout=timeout,
        )

        url_before = page.url
        await locator.click()
        await asyncio.sleep(0.1)

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=3_000)
            except PlaywrightTimeoutError:
                pass
        except PlaywrightTimeoutError:
            pass

        if page.url != url_before:
            stable_count = 0
            last_url = None
            for _ in range(10):
                current_url = page.url
                if current_url == last_url:
                    stable_count += 1
                    if stable_count >= 3:
                        break
                else:
                    stable_count = 0
                    last_url = current_url
                await asyncio.sleep(0.2)

    async def safe_click(self, selector: str, timeout: int = 10_000) -> None:
        """Click with a single retry for transient DOM mutation races."""
        try:
            return await self._execute_click(selector, timeout)
        except PlaywrightError as e:
            message = str(e)
            retryable_fragments = [
                "detached",
                "strict mode",
                "element is not stable",
            ]
            if any(fragment in message for fragment in retryable_fragments):
                logger.warning("safe_click retry triggered for selector '%s': %s", selector, e)
                return await self._execute_click(selector, timeout)
            raise

    async def safe_type(self, selector: str, text: str, timeout: int = 10_000) -> None:
        """Type after strict editable readiness and verify controlled value assignment."""
        page = self.executor._require_page()
        locator = page.locator(selector)

        await locator.wait_for(state="visible", timeout=timeout)
        await locator.wait_for(state="attached", timeout=timeout)
        await page.wait_for_function(
            "(sel) => { const el = document.querySelector(sel); return !!el && !el.disabled && !el.readOnly; }",
            arg=selector,
            timeout=timeout,
        )
        await locator.focus()
        await locator.fill(text)
        await page.wait_for_function(
            "([selector, value]) => document.querySelector(selector)?.value === value",
            arg=[selector, text],
            timeout=timeout,
        )

    async def safe_wait_for_url_contains(self, fragment: str, timeout: int = 10_000) -> None:
        """Wait until current URL contains the expected fragment."""
        page = self.executor._require_page()
        try:
            await page.wait_for_function(
                "(frag) => window.location.href.includes(frag)",
                arg=fragment,
                timeout=timeout,
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Timed out waiting for URL to contain fragment: {fragment}"
            ) from exc

    async def safe_get_text(self, selector: str) -> str:
        """Return stripped inner text for a selector and raise when empty."""
        page = self.executor._require_page()
        locator = page.locator(selector)
        await locator.wait_for(state="visible", timeout=INTERACTION_TIMEOUT_MS)
        text = (await locator.inner_text(timeout=INTERACTION_TIMEOUT_MS)).strip()
        if not text:
            raise RuntimeError(f"Selector has empty text: {selector}")
        return text
