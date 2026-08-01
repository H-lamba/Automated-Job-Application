import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating...")
        await page.goto("https://jobs.ashbyhq.com/openai/25bdc949-e478-4c31-8997-d83482d4c0a2/application")
        await page.wait_for_load_state("networkidle")
        
        inputs = await page.locator("input, textarea").all()
        for i in inputs:
            name = await i.get_attribute("name")
            id = await i.get_attribute("id")
            placeholder = await i.get_attribute("placeholder")
            print(f"Tag: input/textarea, Name: {name}, ID: {id}, Placeholder: {placeholder}")
        
        await browser.close()

asyncio.run(run())
