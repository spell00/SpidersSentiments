"""Inspect Twitter/X reply interface to capture submit button selectors."""

from __future__ import annotations

import logging
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(level=logging.INFO)


def inspect_twitter_reply_interface() -> None:
    """Open Twitter, navigate to a tweet, and inspect the reply interface."""

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Disable Enhanced Tracking Protection to avoid Twitter/X.com blocking
    options.set_preference("privacy.trackingprotection.enabled", False)
    options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
    options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)

    service = Service()
    driver = webdriver.Firefox(service=service, options=options)

    try:
        driver.get("https://x.com")
        driver.add_cookie(
            {
                "name": "auth_token",
                "value": "b8e804c74b35fbfd4c1ec7f7c3aa3cca2cee33de",
                "domain": ".x.com",
            }
        )

        driver.get(
            "https://x.com/search?q=(spider%20OR%20spiders%20OR%20arachnid)%20lang%3Aen%20-is%3Aretweet&src=typed_query&f=live"
        )

        print("🔍 Waiting for page to load...")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))

        articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
        if not articles:
            print("❌ No tweets found")
            return

        print(f"✅ Found {len(articles)} tweets")

        first_tweet = articles[0]
        reply_selectors = [
            "div[data-testid='reply']",
            "button[data-testid='reply']",
            "div[aria-label*='Reply']",
            "button[aria-label*='Reply']",
            "[data-testid='reply']",
        ]

        reply_button = None
        for selector in reply_selectors:
            try:
                reply_button = first_tweet.find_element(By.CSS_SELECTOR, selector)
                print(f"✅ Found reply button with selector: {selector}")
                break
            except Exception:
                continue

        if not reply_button:
            print("❌ Could not find reply button, inspecting first tweet structure:")
            clickable_elements = first_tweet.find_elements(
                By.CSS_SELECTOR,
                "button, div[role='button'], [data-testid*='reply'], [aria-label*='Reply']",
            )
            for element in clickable_elements[:10]:
                print(f"  Element: {element.tag_name}")
                print(f"    data-testid: {element.get_attribute('data-testid')}")
                print(f"    aria-label: {element.get_attribute('aria-label')}")
                print(f"    role: {element.get_attribute('role')}")
                print()
            return

        reply_button.click()

        print("🖱️ Clicked reply button, waiting for compose dialog...")
        time.sleep(2)

        editor_selectors = [
            "div[data-testid='tweetTextarea_0']",
            "div[contenteditable='true']",
            "div[role='textbox']",
        ]

        editor = None
        for selector in editor_selectors:
            try:
                editor = driver.find_element(By.CSS_SELECTOR, selector)
                print(f"✅ Found editor with selector: {selector}")
                break
            except Exception:
                continue

        if not editor:
            print("❌ Could not find text editor")
            return

        editor.send_keys("Test message to check submit buttons")
        print("✅ Typed test message")
        time.sleep(1)

        print("\n🔍 Inspecting submit button candidates:")

        button_selectors = [
            "div[data-testid='tweetButton']",
            "div[data-testid='tweetButtonInline']",
            "button[data-testid='tweetButton']",
            "button[data-testid='tweetButtonInline']",
            "div[data-testid='tweetButtonColored']",
            "button[data-testid='tweetButtonColored']",
            "button[aria-label*='Reply']",
            "button[aria-label*='Tweet']",
            "button[aria-label*='Post']",
            "button:contains('Reply')",
            "button:contains('Tweet')",
            "button:contains('Post')",
        ]

        found_buttons = []
        for selector in button_selectors:
            try:
                if ":contains(" in selector:
                    keyword = selector.split(":contains(")[1].rstrip(")")
                    xpath_selector = f"//button[contains(text(), '{keyword}')]"
                    buttons = driver.find_elements(By.XPATH, xpath_selector)
                else:
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)

                for button in buttons:
                    aria_label = button.get_attribute("aria-label") or ""
                    text_content = button.text or ""
                    disabled = button.get_attribute("aria-disabled") == "true"
                    class_names = button.get_attribute("class") or ""

                    button_info = {
                        "selector": selector,
                        "aria_label": aria_label,
                        "text": text_content,
                        "disabled": disabled,
                        "classes": class_names[:100],
                    }

                    found_buttons.append(button_info)
                    print(f"  🎯 {selector}")
                    print(f"     aria-label: {aria_label}")
                    print(f"     text: {text_content}")
                    print(f"     disabled: {disabled}")
                    print(f"     classes: {class_names[:100]}")
                    print()

            except Exception:
                continue

        if found_buttons:
            print(f"✅ Found {len(found_buttons)} submit button candidates")

            for button_info in found_buttons:
                if button_info["disabled"]:
                    continue
                print(f"🖱️ Attempting to click: {button_info['selector']}")
                try:
                    if ":contains(" in button_info["selector"]:
                        keyword = button_info["selector"].split(":contains(")[1].rstrip(")")
                        xpath_selector = f"//button[contains(text(), '{keyword}')]"
                        button = driver.find_element(By.XPATH, xpath_selector)
                    else:
                        button = driver.find_element(By.CSS_SELECTOR, button_info["selector"])

                    driver.execute_script("arguments[0].click();", button)
                    print("✅ Successfully clicked submit button!")
                    time.sleep(2)
                    break

                except Exception as exc:
                    print(f"❌ Click failed: {exc}")
                    continue
        else:
            print("❌ No submit buttons found")

        with open("twitter_reply_debug.html", "w", encoding="utf-8") as fh:
            fh.write(driver.page_source)
        print("💾 Saved page source to twitter_reply_debug.html")

        print("\n⏱️ Keeping browser open for 30 seconds for manual inspection...")
        time.sleep(30)

    except Exception as exc:  # pragma: no cover - debug helper
        print(f"❌ Error: {exc}")
        try:
            with open("twitter_error_debug.html", "w", encoding="utf-8") as fh:
                fh.write(driver.page_source)
        except Exception:
            pass

    finally:
        driver.quit()


__all__ = ["inspect_twitter_reply_interface"]


if __name__ == "__main__":
    inspect_twitter_reply_interface()
