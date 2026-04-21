import os
from playwright.sync_api import sync_playwright


def test_rag_chatbot_ui():
    app_url = os.getenv("RAG_UI_URL", "http://127.0.0.1:8000/ui")
    prompt = os.getenv(
        "RAG_TEST_PROMPT",
        "What is AI?"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print(f"Opening RAG UI at {app_url} ...")
        page.goto(app_url, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)

        # Ensure chatbot UI loaded.
        assert page.locator("h1", has_text="RAG Chatbot").first.is_visible(), "RAG chatbot header not visible."
        assert page.locator("#query").is_visible(), "Chat input not found."
        assert page.locator("#askBtn").is_visible(), "Send button not found."

        print("Sending test prompt...")
        page.locator("#query").fill(prompt)
        page.locator("#askBtn").click()

        # Wait for at least 2 bot bubbles: greeting + new assistant response.
        page.wait_for_function(
            "() => document.querySelectorAll('#chat .msg.bot').length >= 2",
            timeout=30000,
        )
        page.wait_for_timeout(800)

        bot_messages = page.locator("#chat .msg.bot")
        latest_reply = bot_messages.nth(bot_messages.count() - 1).inner_text().strip()
        assert latest_reply, "Assistant response is empty."
        assert "error:" not in latest_reply.lower(), f"Assistant returned an error: {latest_reply}"

        print("Taking screenshot...")
        page.screenshot(path="rag_chatbot_screenshot.png", full_page=True)
        print("✅ Test passed: Chatbot responded and screenshot captured (rag_chatbot_screenshot.png)")

        browser.close()
        print("All tests completed.")


if __name__ == "__main__":
    test_rag_chatbot_ui()