import asyncio
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from core.config import get_settings
from core.logger import logger


class BrowserAgent:
    def __init__(self):
        self.settings = get_settings()
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None
        
    async def start(self):
        self.playwright = await async_playwright().start()
        
        headless = self.settings.browser.headless
        browser_type = self.settings.browser.browser_type
        
        launch_opts = {
            "headless": headless,
            "args": ["--disable-blink-features=AutomationControlled"]
        }
        
        if browser_type == "chromium":
            self.browser = await self.playwright.chromium.launch(**launch_opts)
        elif browser_type == "firefox":
            self.browser = await self.playwright.firefox.launch(**launch_opts)
        else:
            self.browser = await self.playwright.webkit.launch(**launch_opts)
            
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        self.context = await self.browser.new_context(
            viewport={
                "width": self.settings.browser.viewport_width,
                "height": self.settings.browser.viewport_height,
            },
            user_agent=ua,
        )
        self.page = await self.context.new_page()
        logger.info(
            "Browser Agent started (headless={}, engine={})",
            headless,
            browser_type,
        )
        
    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser Agent closed")
        
    async def navigate(self, url: str) -> bool:
        logger.info(f"Navigating to {url}")
        try:
            await self.page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.settings.browser.timeout * 1000,
            )
            await asyncio.sleep(2)  # Allow React/SPA to hydrate
            return True
        except Exception as e:
            logger.error("Failed to navigate to {}: {}", url, e)
            return False
            
    async def take_screenshot(self, job_id: str, step_name: str) -> str:
        """Takes a screenshot and returns the file path"""
        screenshots_dir = Path(self.settings.storage.screenshots_dir) / job_id
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        path = screenshots_dir / f"{step_name}.png"
        await self.page.screenshot(path=str(path), full_page=True)
        logger.debug(f"Saved screenshot: {path}")
        return str(path)
