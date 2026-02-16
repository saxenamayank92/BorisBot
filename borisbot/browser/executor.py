"""Deterministic Playwright CDP execution layer for browser actions."""

import logging
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

logger = logging.getLogger("borisbot.browser.executor")

NAVIGATION_TIMEOUT_MS = 30_000
INTERACTION_TIMEOUT_MS = 10_000


class BrowserExecutor:
    """Executes deterministic browser actions against a CDP endpoint."""

    def __init__(self, cdp_port: int) -> None:
        self.cdp_port = cdp_port
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def connect(self) -> None:
        """Connect to an existing browser over CDP and prepare an active page."""
        cdp_url = f"http://localhost:{self.cdp_port}"
        logger.info("Connecting to browser CDP endpoint %s", cdp_url)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)

        contexts = self._browser.contexts
        self._context = contexts[0] if contexts else await self._browser.new_context()

        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()

        logger.info("Connected to CDP and ready with active page")

    def _require_page(self) -> Page:
        """Return current page or fail if connect() has not been called."""
        if self._page is None:
            raise RuntimeError("BrowserExecutor is not connected. Call connect() first.")
        return self._page

    async def navigate(self, url: str) -> None:
        """Navigate to URL with a hard 30-second timeout."""
        page = self._require_page()
        logger.info("Navigating to %s", url)
        await page.goto(url, timeout=NAVIGATION_TIMEOUT_MS)

    async def click(self, selector: str) -> None:
        """Click element matching selector with hard 10-second timeout."""
        page = self._require_page()
        logger.info("Clicking selector %s", selector)
        await page.click(selector, timeout=INTERACTION_TIMEOUT_MS)

    async def type(self, selector: str, text: str) -> None:
        """Type text into selector target with hard 10-second timeout."""
        page = self._require_page()
        logger.info("Typing into selector %s", selector)
        await page.fill(selector, "", timeout=INTERACTION_TIMEOUT_MS)
        await page.type(selector, text, timeout=INTERACTION_TIMEOUT_MS)

    async def wait_for(self, selector: str) -> None:
        """Wait for selector with hard 10-second timeout."""
        page = self._require_page()
        logger.info("Waiting for selector %s", selector)
        await page.wait_for_selector(selector, timeout=INTERACTION_TIMEOUT_MS)

    async def get_title(self) -> str:
        """Return current page title."""
        page = self._require_page()
        title = await page.title()
        logger.info("Current page title resolved")
        return title

    async def close(self) -> None:
        """Close browser and Playwright resources."""
        logger.info("Closing browser executor resources")

        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

        self._context = None
        self._page = None
