"""Deterministic browser action primitives built on BrowserExecutor."""

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from borisbot.browser.executor import BrowserExecutor, INTERACTION_TIMEOUT_MS


class BrowserActions:
    """Thin deterministic action layer over BrowserExecutor."""

    def __init__(self, executor: BrowserExecutor):
        self.executor = executor

    async def safe_navigate(self, url: str) -> None:
        """Navigate and verify the page has a non-empty title."""
        await self.executor.navigate(url)
        page = self.executor._require_page()
        await page.wait_for_load_state("networkidle", timeout=INTERACTION_TIMEOUT_MS)
        title = (await self.executor.get_title()).strip()
        if not title:
            raise RuntimeError("Page loaded with empty title")

    async def safe_click(self, selector: str) -> None:
        """Wait for visible element, scroll into view, click, and pause briefly."""
        page = self.executor._require_page()
        locator = page.locator(selector)
        await locator.wait_for(state="visible", timeout=INTERACTION_TIMEOUT_MS)
        await locator.scroll_into_view_if_needed(timeout=INTERACTION_TIMEOUT_MS)
        await locator.click(timeout=INTERACTION_TIMEOUT_MS)
        await page.wait_for_timeout(300)

    async def safe_type(self, selector: str, text: str) -> None:
        """Wait for input, clear it, then type with deterministic delay."""
        page = self.executor._require_page()
        locator = page.locator(selector)
        await locator.wait_for(state="visible", timeout=INTERACTION_TIMEOUT_MS)
        await locator.fill("", timeout=INTERACTION_TIMEOUT_MS)
        await locator.type(text, delay=50, timeout=INTERACTION_TIMEOUT_MS)

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
