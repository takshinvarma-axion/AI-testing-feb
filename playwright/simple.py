from playwright.sync_api import sync_playwright

def simple_browser_demo(url="https://google.com"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://google.com", wait_until="domcontentloaded")
        print("Page title:", page.title())
        page.screenshot(path="playwright_example.png", full_page=True)
        browser.close()

simple_browser_demo()