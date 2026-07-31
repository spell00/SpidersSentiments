"""Twitter/X Selenium client for Spider Guardian."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

from .config import SpiderGuardianConfig


@dataclass
class SocialPost:
    id: str
    text: str
    conversation_id: str
    lang: str = "en"
    author_handle: Optional[str] = None
    is_reply: bool = False
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    impression_count: int = 0
    image_urls: List[str] = field(default_factory=list)


class SeleniumTwitterClient:
    RATE_LIMIT_WAIT_PATH = "data/selenium_wait_time.json"
    RATE_LIMIT_WAIT_DEFAULT = 120
    RATE_LIMIT_WAIT_MAX = 1800
    BASE_URL = "https://x.com"

    def __init__(self, config: SpiderGuardianConfig) -> None:
        self.config = config
        self.username = os.getenv("TWITTER_USERNAME")
        self.password = os.getenv("TWITTER_PASSWORD")
        self.auth_cookie = os.getenv("TWITTER_AUTH_COOKIE")
        self.ct0 = os.getenv("TWITTER_CT0", "")
        if not self.auth_cookie and (not self.username or not self.password):
            raise RuntimeError(
                "Set TWITTER_USERNAME/TWITTER_PASSWORD or TWITTER_AUTH_COOKIE for Selenium transport."
            )

        self.driver = self._build_driver(headless=config.selenium_headless, driver_name=config.selenium_driver)
        self.wait = WebDriverWait(self.driver, config.selenium_wait_seconds)
        atexit.register(self.close)

        if self.auth_cookie:
            self._inject_cookie()
        else:
            self._login_with_credentials()

    # --- driver helpers -------------------------------------------------

    def _build_driver(self, headless: bool, driver_name: str):
        """Build a Selenium WebDriver with robust fallbacks.

        - Supports FIREFOX_BINARY env var to point to custom Firefox installation
        - Falls back to Chrome if Firefox fails to start
        - Adds clearer logging for troubleshooting
        """
        if driver_name == "firefox":
            try:
                options = FirefoxOptions()
                if headless:
                    options.add_argument("-headless")
                # Allow overriding Firefox binary location via env var
                firefox_bin = os.getenv("FIREFOX_BINARY")
                if firefox_bin:
                    try:
                        options.binary_location = firefox_bin
                        logging.info("Using custom Firefox binary: %s", firefox_bin)
                    except Exception as e:
                        logging.warning("Failed to set Firefox binary '%s': %s", firefox_bin, e)

                # Disable Enhanced Tracking Protection to prevent blocking Twitter/X.com
                options.set_preference("privacy.trackingprotection.enabled", False)
                options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
                options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)
                options.set_preference("privacy.trackingprotection.fingerprinting.enabled", False)
                options.set_preference("privacy.trackingprotection.cryptomining.enabled", False)

                gecko_path = GeckoDriverManager().install()
                logging.info("GeckoDriver path: %s", gecko_path)
                service = FirefoxService(executable_path=gecko_path)
                return webdriver.Firefox(service=service, options=options)
            except Exception as exc:
                logging.error("Firefox WebDriver failed to start: %s", exc)
                logging.info("Falling back to Chrome WebDriver. You can also run with --selenium-driver chrome")
                # Fall through to Chrome fallback

        # Chrome (default or fallback)
        options = ChromeOptions()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        if headless:
            options.add_argument("--headless=new")
        chrome_path = ChromeDriverManager().install()
        logging.info("ChromeDriver path: %s", chrome_path)
        service = ChromeService(chrome_path)
        return webdriver.Chrome(service=service, options=options)

    def _inject_cookie(self) -> None:
        for domain in ("x.com", "twitter.com"):
            try:
                url = f"https://{domain}"
                self.driver.get(url)
                self.driver.add_cookie({"name": "auth_token", "value": self.auth_cookie, "domain": domain, "path": "/"})
                if self.ct0:
                    self.driver.add_cookie({"name": "ct0", "value": self.ct0, "domain": domain, "path": "/"})
            except Exception:
                continue
        self.driver.get(self.BASE_URL)
        self._wait_primary_column()
        self._best_effort_handle_detection()

    def _login_with_credentials(self) -> None:  # pragma: no cover - interactive automation
        self.driver.get(f"{self.BASE_URL}/login")
        time.sleep(2.0)
        username_field = self.wait.until(EC.presence_of_element_located((By.NAME, "text")))
        username_field.clear()
        username_field.send_keys(self.username)
        username_field.send_keys(Keys.RETURN)
        time.sleep(1.5)
        password_field = self.wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_field.clear()
        password_field.send_keys(self.password)
        password_field.send_keys(Keys.RETURN)
        self._wait_primary_column()
        self._best_effort_handle_detection()

    def _best_effort_handle_detection(self) -> None:
        with contextlib.suppress(Exception):
            anchors = self.driver.find_elements(By.CSS_SELECTOR, "a[href^='https://x.com/']")
            for anchor in anchors:
                href = (anchor.get_attribute("href") or "").rstrip("/")
                parts = href.split("/")
                if len(parts) >= 4:
                    handle = parts[3]
                    if handle and handle.lower() not in {
                        "home",
                        "explore",
                        "notifications",
                        "messages",
                        "settings",
                    }:
                        self.username = self.username or handle
                        logging.info("[AUTH] Detected logged-in handle=%s", self.username)
                        break

    def _wait_primary_column(self) -> None:
        with contextlib.suppress(TimeoutException):
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='primaryColumn']")))
            return
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "main[role='main']")))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.driver.quit()

    # --- rate limit persistence ----------------------------------------

    def _load_rate_limit_wait(self) -> int:
        try:
            with open(self.RATE_LIMIT_WAIT_PATH, "r", encoding="utf-8") as handle:
                val = json.load(handle)
            return int(val)
        except Exception:
            return self.RATE_LIMIT_WAIT_DEFAULT

    def _save_rate_limit_wait(self, wait_time: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.RATE_LIMIT_WAIT_PATH), exist_ok=True)
            with open(self.RATE_LIMIT_WAIT_PATH, "w", encoding="utf-8") as handle:
                json.dump(wait_time, handle)
        except Exception:
            pass

    # --- rate limit persistence ----------------------------------------

    # --- collection -----------------------------------------------------

    def search_posts(self, query: str, mode: str = "live") -> List[SocialPost]:  # pragma: no cover - Selenium heavy
        encoded = urllib.parse.quote(query)
        mode = mode if mode in {"live", "top", "latest"} else "live"
        url = f"{self.BASE_URL}/search?q={encoded}&f={mode}"
        wait_time = self._load_rate_limit_wait()
        for attempt in range(3):
            self.driver.get(url)
            self._wait_primary_column()
            if "something went wrong" in (self.driver.page_source or "").lower():
                logging.warning("[search_posts] Rate-limit page encountered. Waiting %ss", wait_time)
                self.driver.get(f"{self.BASE_URL}/home")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, self.RATE_LIMIT_WAIT_MAX)
                self._save_rate_limit_wait(wait_time)
                continue

            collected: List[SocialPost] = []
            seen: set[str] = set()
            for _ in range(8):
                try:
                    cards = self._wait_for_cards(min_cards=1)
                except TimeoutException as exc:
                    logging.warning("[search_posts] Could not find cards: %s", exc)
                    time.sleep(1.5)
                    continue
                for card in cards:
                    post = self._extract_post(card)
                    if not post or post.id in seen:
                        continue
                    seen.add(post.id)
                    collected.append(post)
                    logging.info(
                        "[search_posts] Collected post with ID %s and metrics: likes=%d, reposts=%d, replies=%d, impressions=%d", 
                        post.id, post.like_count, post.repost_count, post.reply_count, post.impression_count
                    )
                with contextlib.suppress(Exception):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.2)
            if collected:
                self._save_rate_limit_wait(self.RATE_LIMIT_WAIT_DEFAULT)
                return collected
            logging.info("[search_posts] No cards found, retrying after %ss", wait_time)
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, self.RATE_LIMIT_WAIT_MAX)
            self._save_rate_limit_wait(wait_time)
        logging.error("[search_posts] Failed after retries")
        return []

    # --- posting --------------------------------------------------------

    def reply(self, text: str, reply_to_tweet_id: str) -> Optional[str]:  # pragma: no cover
        wait_time = self._load_rate_limit_wait()
        url = f"{self.BASE_URL}/i/status/{reply_to_tweet_id}"
        # Reduced timeout to prevent hanging - 30s should be enough
        overall_deadline = time.time() + 30
        
        logging.info("[reply] 🚀 Starting reply to %s: %r (30s timeout)", reply_to_tweet_id, text[:50])
        
        # Modify the reply text to keep only the part after 'reply:'
        if text.lower().startswith('reply:'):
            text = text.split(':', 1)[1].strip()
            logging.info("[reply] Modified text to keep content after 'reply:'")
        
        for attempt in range(3):
            if time.time() > overall_deadline:
                logging.warning("[reply] ⏰ Overall deadline exceeded, aborting")
                break
                
            attempt_start = time.time()
            logging.info("[reply] 🔄 Attempt %d/3: Loading tweet page", attempt + 1)
            
            try:
                self.driver.get(url)
                self._wait_primary_column()
            except Exception as exc:
                logging.warning("[reply] Failed to load page: %s", exc)
                continue
            if "something went wrong" in (self.driver.page_source or "").lower():
                logging.warning("[reply] Rate-limit page encountered. Waiting %ss", wait_time)
                self.driver.get(f"{self.BASE_URL}/home")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, self.RATE_LIMIT_WAIT_MAX)
                self._save_rate_limit_wait(wait_time)
                continue
            try:
                card = self._wait_for_cards(min_cards=1)[0]
            except TimeoutException as exc:
                logging.warning("[reply] Could not find cards: %s", exc)
                continue
            with contextlib.suppress(Exception):
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", card)

            # Try to open composer and obtain editor within a small deadline.
            editor = None
            open_deadline = time.time() + 12
            composer_attempts = 0
            max_composer_attempts = 8
            
            while time.time() < open_deadline and editor is None and composer_attempts < max_composer_attempts:
                composer_attempts += 1
                logging.debug("[reply] Composer attempt %d/%d", composer_attempts, max_composer_attempts)
                
                # Try to open the composer
                composer_opened = self._open_reply_composer(card)
                if not composer_opened:
                    logging.debug("[reply] Composer click failed, trying keyboard shortcut")
                    # Try key shortcut fallback
                    try:
                        ActionChains(self.driver).move_to_element(card).click().send_keys("r").perform()
                        time.sleep(0.3)
                    except Exception as exc:
                        logging.debug("[reply] Keyboard shortcut failed: %s", exc)
                
                # Some flows show a blocking 'Done' overlay; try to dismiss it
                self._click_done_or_close_overlays()
                
                # Try to find the editor
                editor = self._locate_reply_editor()
                
                if editor is None:
                    logging.debug("[reply] Editor not found, dismissing modals and retrying")
                    # Dismiss stray modals if any and retry
                    try:
                        ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                        time.sleep(0.3)
                    except Exception:
                        pass
                else:
                    logging.debug("[reply] ✅ Editor found after %d attempts", composer_attempts)
                    break

            if editor is None:
                logging.warning("[reply] Editor not found after retries; skipping tweet %s", reply_to_tweet_id)
                continue

            if not self._send_reply_text(editor, text):
                logging.warning("[reply] Failed to type into editor; skipping")
                # Try to close the composer so we don't get stuck
                with contextlib.suppress(Exception):
                    ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                continue

            # Introduce a delay to mimic human-like behavior
            delay_seconds = 5  # Adjust the delay as needed
            logging.info(f"[reply] Waiting for {delay_seconds} seconds before submitting the reply...")
            time.sleep(delay_seconds)

            # Check for timeout before proceeding to submit
            if time.time() > overall_deadline:
                logging.warning("[reply] ⏰ Timeout before submit phase")
                break
                
            logging.info("[reply] Text entered successfully. Checking for overlays...")
            # If a 'Done' button is required before posting, click it
            # self._click_done_or_close_overlays()

            submit_start = time.time()
            logging.info("[reply] Attempting to submit reply...")
            
            if self._click_reply_submit():
                submit_time = time.time() - submit_start
                logging.info("[reply] Submit clicked (%.1fs). Looking for reply ID...", submit_time)
                
                # Much shorter timeout for ID lookup to prevent hanging
                id_lookup_start = time.time()
                reply_id = self._locate_latest_reply_id(expected_text=text)
                id_lookup_time = time.time() - id_lookup_start
                
                if reply_id:
                    total_time = time.time() - attempt_start
                    logging.info("[reply] ✅ Reply successful! ID: %s (total: %.1fs, submit: %.1fs, lookup: %.1fs)", 
                               reply_id, total_time, submit_time, id_lookup_time)
                    self._save_rate_limit_wait(self.RATE_LIMIT_WAIT_DEFAULT)
                    return reply_id
                else:
                    logging.warning("[reply] ⚠️ Submit succeeded but no ID found after %.1fs - reply may still have posted", id_lookup_time)
                    # Return a placeholder to indicate likely success
                    return "posted_no_id_found"
            else:
                logging.warning("[reply] Submit button click failed after %.1fs", time.time() - submit_start)
                # Close composer and retry next attempt
                try:
                    ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                    time.sleep(0.2)
                except Exception:
                    pass
        # Emergency cleanup - close any open composers
        logging.error("[reply] ❌ Failed to send reply after all attempts")
        try:
            # Try to close any open modals/composers
            for _ in range(3):
                ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(0.2)
            # Navigate back to home to reset state
            self.driver.get(f"{self.BASE_URL}/home")
        except Exception as cleanup_exc:
            logging.warning("[reply] Emergency cleanup failed: %s", cleanup_exc)
        
        return None

    def post_tweet(self, text: str) -> Optional[str]:  # pragma: no cover - Selenium heavy
        """Publish a brand-new tweet using the global composer."""

        payload = (text or "").strip()
        if not payload:
            raise ValueError("Cannot post empty text")

        logging.info("[post] 🚀 Publishing original post: %r", payload[:60])

        for attempt in range(3):
            try:
                logging.debug("[post] Attempt %d/3: opening compose surface", attempt + 1)
                self.driver.get(f"{self.BASE_URL}/compose/post")
                composer_wait = WebDriverWait(self.driver, 15)
                composer_wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox'][contenteditable='true']"))
                )

                editor = self._locate_reply_editor()
                if editor is None:
                    logging.warning("[post] Composer editor unavailable on attempt %d", attempt + 1)
                    self._click_done_or_close_overlays()
                    continue

                if not self._send_reply_text(editor, payload):
                    logging.warning("[post] Unable to inject text on attempt %d", attempt + 1)
                    self._click_done_or_close_overlays()
                    continue

                if not self._click_reply_submit():
                    logging.warning("[post] Submit button not reachable on attempt %d", attempt + 1)
                    self._click_done_or_close_overlays()
                    continue

                confirmed = self._await_post_confirmation()
                tweet_id = self._extract_tweet_id_from_url(self.driver.current_url)

                if tweet_id:
                    logging.info("[post] ✅ Post live at %s", tweet_id)
                else:
                    logging.info("[post] ✅ Post submitted%s", " (no toast detected)" if not confirmed else "")

                if "/compose" in (self.driver.current_url or ""):
                    with contextlib.suppress(Exception):
                        self.driver.get(self.BASE_URL)

                return tweet_id
            except Exception as exc:
                logging.warning("[post] Attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2.0)

        logging.error("[post] ❌ Unable to publish post after retries")
        return None

    # --- scrolling helpers ---------------------------------------------

    def _wait_for_cards(self, min_cards: int = 1):
        selectors = [
            "article[data-testid='tweet']",
            "article[data-testid='post']",
            "article[data-testid]",
            "article:has(div[data-testid='tweetText'])",
            "article[data-testid='cellInnerDiv']",
        ]
        last_exc: Optional[Exception] = None
        for selector in selectors:
            try:
                cards = self.wait.until(
                    lambda drv: (elems := drv.find_elements(By.CSS_SELECTOR, selector))
                    and len(elems) >= min_cards
                    and elems
                )
                if cards:
                    return cards
            except Exception as exc:
                last_exc = exc
                with contextlib.suppress(Exception):
                    self.driver.execute_script("window.scrollBy(0, 400);")
        if last_exc:
            raise TimeoutException(str(last_exc))
        raise TimeoutException("No tweet cards located")

    def _extract_post(self, card, conversation_id: Optional[str] = None) -> Optional[SocialPost]:
        try:
            link = card.find_element(By.XPATH, ".//a[contains(@href, '/status/')][last()]")
            href = link.get_attribute("href")
            if not href:
                return None
            post_id = href.rstrip("/").split("/")[-1]
            text_nodes = card.find_elements(By.CSS_SELECTOR, "div[data-testid='tweetText']")
            body = " ".join(n.text for n in text_nodes).strip()
            author = None
            # Primary selector (most reliable on X.com today)
            with contextlib.suppress(Exception):
                author_link = card.find_element(By.CSS_SELECTOR, "div[data-testid='User-Name'] a[href*='x.com/']")
                ahref = author_link.get_attribute("href")
                if ahref:
                    author = ahref.rstrip("/").split("/")[-1]
            # Fallback 1: any profile link inside the card that is not a status link
            if not author:
                with contextlib.suppress(Exception):
                    alts = card.find_elements(By.XPATH, ".//a[starts-with(@href,'https://x.com/') and not(contains(@href,'/status/'))]")
                    for a in alts:
                        href = (a.get_attribute("href") or "").rstrip("/")
                        parts = href.split("/")
                        if len(parts) >= 4:
                            candidate = parts[3]
                            if candidate and candidate.lower() not in {"home","explore","notifications","messages","settings"}:
                                author = candidate
                                break
            # Fallback 2: look for a span with @handle in the user name block
            if not author:
                with contextlib.suppress(Exception):
                    span = card.find_element(By.XPATH, ".//div[@data-testid='User-Name']//span[starts-with(normalize-space(text()), '@')]")
                    handle_text = (span.text or "").strip()
                    if handle_text.startswith("@") and 1 < len(handle_text) <= 17:
                        author = handle_text[1:]
            # Fallback 3: generic anchors with @text
            if not author:
                with contextlib.suppress(Exception):
                    spans = card.find_elements(By.XPATH, ".//span[starts-with(normalize-space(text()), '@')]")
                    for sp in spans[:3]:
                        t = (sp.text or "").strip()
                        if t.startswith("@") and 1 < len(t) <= 17:
                            author = t[1:]
                            break
            is_reply = self._card_is_reply(card)
            reply_c, repost_c, like_c, impression_c = self._extract_metrics(card)
            
            # Extract image URLs
            image_urls = self._extract_image_urls(card)
            
            return SocialPost(
                id=post_id,
                text=body or "",
                conversation_id=conversation_id or post_id,
                author_handle=author,
                is_reply=is_reply,
                like_count=like_c,
                repost_count=repost_c,
                reply_count=reply_c,
                impression_count=impression_c,
                image_urls=image_urls,
            )
        except (NoSuchElementException, StaleElementReferenceException):
            return None

    def _parse_count(self, raw: str) -> int:
        s = (raw or "").strip().lower().replace(",", "")
        try:
            if s.endswith("k"):
                return int(float(s[:-1]) * 1_000)
            if s.endswith("m"):
                return int(float(s[:-1]) * 1_000_000)
            return int(float(s))
        except Exception:
            # Attempt to extract leading number from phrases like "12 Likes"
            num = ""
            for ch in s:
                if ch.isdigit() or ch in ".,":
                    num += ch
                elif num:
                    break
            num = num.replace(",", "")
            with contextlib.suppress(Exception):
                return int(float(num))
            return 0

    def _extract_metrics(self, card) -> tuple[int, int, int, int]:
        reply_c = repost_c = like_c = impression_c = 0
        # Try aria-labels on buttons first
        mapping = {
            "reply": "reply",
            "retweet": "repost",
            "like": "like",
            "impression": "impression",
        }
        for testid, key in mapping.items():
            with contextlib.suppress(Exception):
                btns = card.find_elements(By.CSS_SELECTOR, f"div[data-testid='{testid}'], button[data-testid='{testid}']")
                for btn in btns:
                    aria = (btn.get_attribute("aria-label") or btn.get_attribute("title") or "").lower()
                    if not aria:
                        continue
                    val = self._parse_count(aria)
                    if key == "reply":
                        reply_c = max(reply_c, val)
                    elif key == "repost":
                        repost_c = max(repost_c, val)
                    elif key == "like":
                        like_c = max(like_c, val)
                    elif key == "impression":
                        impression_c = max(impression_c, val)
        # Fallback: numbers near the icons
        try:
            containers = card.find_elements(By.CSS_SELECTOR, "div[data-testid='reply'], div[data-testid='retweet'], div[data-testid='like'], div[data-testid='impression']")
            for cont in containers:
                label = (cont.get_attribute('data-testid') or '').lower()
                with contextlib.suppress(Exception):
                    span = cont.find_element(By.XPATH, ".//span[normalize-space(text())!='']")
                    val = self._parse_count(span.text)
                    if 'reply' in label:
                        reply_c = max(reply_c, val)
                    elif 'retweet' in label:
                        repost_c = max(repost_c, val)
                    elif 'like' in label:
                        like_c = max(like_c, val)
                    elif 'impression' in label:
                        impression_c = max(impression_c, val)
        except Exception:
            pass
        return reply_c, repost_c, like_c, impression_c

    def resolve_author_handle(self, post_id: str) -> Optional[str]:
        """Navigate to a tweet page and extract the author's handle.
        
        This is slower than parsing the search card, so it's used as a fallback
        when author_handle isn't present in search results.
        """
        try:
            url = f"{self.BASE_URL}/i/status/{post_id}"
            self.driver.get(url)
            self._wait_primary_column()
            # Primary selector
            with contextlib.suppress(Exception):
                author_link = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='User-Name'] a[href*='x.com/']")
                ahref = author_link.get_attribute("href")
                if ahref:
                    return ahref.rstrip("/").split("/")[-1]
            # Fallbacks on the page
            with contextlib.suppress(Exception):
                span = self.driver.find_element(By.XPATH, "//div[@data-testid='User-Name']//span[starts-with(normalize-space(text()), '@')]")
                t = (span.text or "").strip()
                if t.startswith("@") and 1 < len(t) <= 17:
                    return t[1:]
            with contextlib.suppress(Exception):
                anchors = self.driver.find_elements(By.XPATH, "//a[starts-with(@href,'https://x.com/') and not(contains(@href,'/status/'))]")
                for a in anchors:
                    href = (a.get_attribute('href') or '').rstrip('/')
                    parts = href.split('/')
                    if len(parts) >= 4:
                        user = parts[3]
                        if user and user.lower() not in {"home","explore","notifications","messages","settings"}:
                            return user
        except Exception as exc:
            logging.debug("[resolve_author_handle] Failed to resolve handle for %s: %s", post_id, exc)
        return None

    def extract_views_from_page(self) -> int:
        """Extract view count from current page using the same regex approach as trending uploads.
        
        X/Twitter displays Views in the page text, not in a structured data-testid attribute.
        This matches the proven pattern from langsmith/config.py for trending posts.
        """
        import re
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?[KMB]?)\s+Views?", page_text, re.IGNORECASE)
            if match:
                raw_count = match.group(1)
                views = self._parse_count(raw_count)
                logging.debug("[views] Extracted %d views from page text: %s", views, raw_count)
                return views
        except Exception as exc:
            logging.debug("[views] Failed to extract views from page: %s", exc)
        return 0

    def _extract_image_urls(self, card) -> List[str]:
        """Extract image URLs from a tweet card"""
        image_urls = []
        try:
            # Look for images in the tweet
            img_elements = card.find_elements(By.CSS_SELECTOR, "img[src*='pbs.twimg.com']")
            for img in img_elements:
                src = img.get_attribute("src")
                if src and "pbs.twimg.com" in src:
                    # Convert to high quality version
                    if "?format=" in src:
                        # Remove size modifiers to get original quality
                        base_url = src.split("?")[0]
                        image_urls.append(f"{base_url}?format=jpg&name=large")
                    else:
                        image_urls.append(src)
        except Exception as e:
            logging.debug(f"Failed to extract image URLs: {e}")
        
        return list(set(image_urls))  # Remove duplicates

    def _card_is_reply(self, card) -> bool:
        with contextlib.suppress(Exception):
            if card.find_elements(By.XPATH, ".//div[@data-testid='reply']"):
                return True
        with contextlib.suppress(Exception):
            contexts = card.find_elements(By.CSS_SELECTOR, "div[data-testid='tweetContext']")
            for ctx in contexts:
                if "replying to" in (ctx.text or "").lower():
                    return True
        with contextlib.suppress(Exception):
            aria_label = (card.get_attribute("aria-label") or "").lower()
            if "replying to" in aria_label:
                return True
        with contextlib.suppress(Exception):
            spans = card.find_elements(
                By.XPATH,
                ".//span[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'replying to')]",
            )
            if spans:
                return True
        with contextlib.suppress(Exception):
            full_text = (card.text or "").lower()
            if "replying to" in full_text:
                return True
        return False

    def _open_reply_composer(self, card) -> bool:
        candidates = [
            (By.CSS_SELECTOR, "button[data-testid='reply']"),  # Prioritize the working button selector
            (By.CSS_SELECTOR, "div[data-testid='reply']"),     # Fallback to div version
            (By.XPATH, "//article//div[@data-testid='reply']/ancestor::*[@role='button'][1]"),
            (
                By.XPATH,
                "//article//*[@role='button'][@aria-label and contains(translate(@aria-label,'REPLY','reply'),'reply')]",
            ),
        ]
        for by, selector in candidates:
            try:
                element = self.wait.until(EC.element_to_be_clickable((by, selector)))
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
                element.click()
                return True
            except Exception:
                continue
        with contextlib.suppress(Exception):
            ActionChains(self.driver).move_to_element(card).click().send_keys("r").perform()
            return True
        return False

    def _locate_reply_editor(self):
        selectors = [
            # Most specific, most likely to be reply editor
            (By.CSS_SELECTOR, "div[role='textbox'][data-testid^='tweetTextarea_Reply']"),
            (By.CSS_SELECTOR, "div[role='textbox'][data-testid^='tweetTextarea']"),
            (By.CSS_SELECTOR, "div[role='textbox'][contenteditable='true'][aria-label*='Reply']"),
            (By.CSS_SELECTOR, "div[role='textbox'][contenteditable='true']"),
            (By.XPATH, "//div[@role='textbox' and @contenteditable='true' and contains(@aria-label, 'Reply') ]"),
            (By.XPATH, "//div[@role='textbox' and @contenteditable='true']"),
            (By.CSS_SELECTOR, "div[aria-multiline='true'][role='textbox']"),
            (By.XPATH, "//div[contains(@data-testid,'tweetTextarea') and @role='textbox']"),
            (By.XPATH, "//div[@role='textbox']"),
        ]

        for i, (by, selector) in enumerate(selectors):
            try:
                # Use a slightly longer wait for each selector attempt
                short_wait = WebDriverWait(self.driver, 3)
                editor = short_wait.until(EC.presence_of_element_located((by, selector)))

                # Wait for visibility and interactability
                WebDriverWait(self.driver, 2).until(EC.visibility_of(editor))
                WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((by, selector)))

                # Verify the editor is actually interactable
                if not editor.is_displayed():
                    continue

                # Check if it's the right editor (not a search box or other textbox)
                parent_html = self.driver.execute_script("return arguments[0].parentElement.innerHTML;", editor)
                if "search" in parent_html.lower() or "message" in parent_html.lower():
                    continue

                # Ensure it is visible and interactable
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", editor)

                # Additional check - make sure we can focus it
                try:
                    self.driver.execute_script("arguments[0].focus();", editor)
                    # If we got here, this editor looks good
                    logging.debug("[reply] Found editor with selector %d: %s", i + 1, selector)
                    return editor
                except Exception as focus_exc:
                    logging.debug("[reply] Focus failed for selector %d: %s", i + 1, focus_exc)
                    continue

            except Exception as exc:
                logging.debug("[reply] Selector %d failed: %s", i + 1, exc)
                continue

        # Fallback diagnostics: screenshot and HTML dump
        try:
            ts = int(time.time())
            screenshot_path = f"logs/reply_editor_not_found_{ts}.png"
            html_path = f"logs/reply_editor_not_found_{ts}.html"
            self.driver.save_screenshot(screenshot_path)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logging.warning(f"[reply] Reply editor not found after trying {len(selectors)} selectors. Screenshot: {screenshot_path}, HTML: {html_path}")
        except Exception as diag_exc:
            logging.warning(f"[reply] Diagnostics failed: {diag_exc}")
        return None

    def _send_reply_text(self, editor, text: str) -> bool:
        # Try multiple focus/type strategies; bail out if all fail
        typing_deadline = time.time() + 15  # Hard timeout to prevent hanging
        
        for step in range(4):
            if time.time() > typing_deadline:
                logging.warning("[reply] Typing timeout reached, aborting")
                return False
                
            step_start = time.time()
            logging.debug("[reply] Typing attempt %d/4", step + 1)
            
            # Focus the editor with timeout
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", editor)
                ActionChains(self.driver).move_to_element(editor).click().perform()
            except Exception as exc:
                logging.debug("[reply] Focus click failed: %s", exc)
                try:
                    self.driver.execute_script("arguments[0].focus();", editor)
                except Exception as exc2:
                    logging.debug("[reply] JS focus failed: %s", exc2)
            
            time.sleep(0.2)
            
            # Method 1: Native send_keys with clearing
            try:
                # Clear existing text
                ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("a").key_up(Keys.CONTROL).send_keys(Keys.BACKSPACE).perform()
                time.sleep(0.1)
                editor.send_keys(text)
                
                # Verify text was entered by checking element content
                current_text = self.driver.execute_script("return arguments[0].textContent || arguments[0].value || '';", editor)
                if text.strip() in current_text.strip():
                    logging.debug("[reply] ✅ Native send_keys successful (%.1fs)", time.time() - step_start)
                    return True
                else:
                    logging.debug("[reply] Native send_keys failed - text not found in editor")
            except Exception as exc:
                logging.debug("[reply] Native send_keys failed: %s", exc)
            
            # Method 2: JavaScript injection with verification
            try:
                result = self.driver.execute_script(
                    """
                    const el = arguments[0];
                    const txt = arguments[1];
                    try {
                        if (el.isContentEditable) {
                            el.focus();
                            el.textContent = txt;
                            const ev = new InputEvent('input', {bubbles:true, cancelable:true});
                            el.dispatchEvent(ev);
                            return el.textContent;
                        } else if (el.value !== undefined) {
                            el.focus();
                            el.value = txt;
                            el.dispatchEvent(new Event('input', {bubbles:true}));
                            return el.value;
                        }
                        return null;
                    } catch (e) {
                        return 'error: ' + e.message;
                    }
                    """,
                    editor,
                    text,
                )
                
                if result and text.strip() in str(result).strip():
                    # Nudge React by simulating additional input
                    try:
                        editor.send_keys(" ")
                        ActionChains(self.driver).send_keys(Keys.BACKSPACE).perform()
                    except Exception:
                        pass
                    logging.debug("[reply] ✅ JS injection successful (%.1fs)", time.time() - step_start)
                    return True
                else:
                    logging.debug("[reply] JS injection failed - result: %s", result)
            except Exception as exc:
                logging.debug("[reply] JS injection failed: %s", exc)
            
            # Method 3: Character-by-character typing (last resort)
            if step == 3:  # Only on final attempt
                try:
                    ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("a").key_up(Keys.CONTROL).perform()
                    time.sleep(0.1)
                    for char in text:
                        if time.time() > typing_deadline:
                            break
                        editor.send_keys(char)
                        time.sleep(0.01)  # Small delay between characters
                    
                    current_text = self.driver.execute_script("return arguments[0].textContent || arguments[0].value || '';", editor)
                    if text.strip() in current_text.strip():
                        logging.debug("[reply] ✅ Character typing successful (%.1fs)", time.time() - step_start)
                        return True
                except Exception as exc:
                    logging.debug("[reply] Character typing failed: %s", exc)
            
            time.sleep(0.3)
        
        logging.warning("[reply] All editor typing strategies failed after %.1fs", time.time() - typing_deadline + 15)
        return False

    def _click_reply_submit(self) -> bool:
        submit_deadline = time.time() + 8  # Hard timeout for submit
        
        candidates = [
            (By.CSS_SELECTOR, "div[data-testid='tweetButton'], div[data-testid='tweetButtonInline']"),
            (By.CSS_SELECTOR, "button[data-testid='tweetButton'], button[data-testid='tweetButtonInline']"),
            (By.CSS_SELECTOR, "div[data-testid='tweetButtonColored'], button[data-testid='tweetButtonColored']"),
            (By.CSS_SELECTOR, "button[aria-label$='Reply']"),
            (By.XPATH, "//button[contains(@aria-label,'Reply')]") ,
            (By.XPATH, "//button[contains(.,'Reply') or contains(.,'Tweet') or contains(.,'Send')]")
        ]
        
        for i, (by, selector) in enumerate(candidates):
            if time.time() > submit_deadline:
                logging.warning("[reply] Submit deadline exceeded at selector %d", i)
                break
                
            try:
                # Use shorter wait for each button attempt
                short_wait = WebDriverWait(self.driver, 2)
                button = short_wait.until(EC.element_to_be_clickable((by, selector)))
                
                aria_disabled = button.get_attribute("aria-disabled")
                disabled = button.get_attribute("disabled")
                
                # Check if button is actually clickable
                if aria_disabled == "true" or disabled == "true":
                    logging.debug("[reply] Button %d disabled, trying next", i)
                    continue
                
                logging.debug("[reply] Clicking submit button %d: aria-disabled=%s disabled=%s", i, aria_disabled, disabled)
                
                # Try clicking with JS if regular click might hang
                try:
                    self.driver.execute_script("arguments[0].click();", button)
                    logging.debug("[reply] ✅ Submit button clicked via JS")
                    return True
                except Exception:
                    # Fallback to regular click
                    button.click()
                    logging.debug("[reply] ✅ Submit button clicked normally")
                    return True
                    
            except Exception as exc:
                logging.debug("[reply] Submit button %d failed: %s", i, exc)
                continue
        
        # Last resort: keyboard shortcut with timeout
        if time.time() <= submit_deadline:
            try:
                logging.debug("[reply] Trying keyboard shortcut as fallback")
                ActionChains(self.driver).key_down(Keys.CONTROL).send_keys(Keys.ENTER).key_up(Keys.CONTROL).perform()
                time.sleep(0.5)  # Brief wait to see if it worked
                return True
            except Exception as exc:
                logging.debug("[reply] Keyboard shortcut failed: %s", exc)
        
        logging.error("[reply] ❌ Could not activate send button after %.1fs", time.time() - submit_deadline + 8)
        return False

    def _await_post_confirmation(self, timeout: float = 10.0) -> bool:
        """Wait for the platform toast that indicates the post was sent."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            with contextlib.suppress(Exception):
                alert = self.driver.find_element(By.CSS_SELECTOR, "div[role='alert']")
                text = (alert.text or "").lower()
                if any(token in text for token in ("sent", "posted", "published")):
                    return True
            time.sleep(0.4)
        return False

    @staticmethod
    def _extract_tweet_id_from_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        if "/status/" not in url:
            return None
        return url.rstrip("/").split("/")[-1]

    # --- overlay helpers ---------------------------------------------

    def _click_done_or_close_overlays(self) -> bool:
        """Try to click 'Done' or close common overlay dialogs.

        Returns True if something was clicked.
        """
        clicked = False
        selectors = [
            (By.XPATH, "//button[normalize-space()='Done']"),
            (By.XPATH, "//button[contains(.,'Done')]"),
            (By.XPATH, "//div[@role='button' and (normalize-space()='Done' or contains(.,'Done'))]"),
            (By.CSS_SELECTOR, "button[aria-label='Done'], button[aria-label*='Done']"),
        ]
        for by, sel in selectors:
            with contextlib.suppress(Exception):
                btns = self.driver.find_elements(by, sel)
                for btn in btns:
                    if not btn.is_displayed():
                        continue
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    btn.click()
                    logging.info("[reply] Clicked 'Done' overlay button")
                    time.sleep(0.3)
                    clicked = True
                    break
            if clicked:
                break
        if not clicked:
            with contextlib.suppress(Exception):
                ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                clicked = True
        return clicked

    def _locate_latest_reply_id(self, expected_text: Optional[str] = None) -> Optional[str]:
        normalized_expected = " ".join((expected_text or "").split())
        deadline = time.time() + 5  # Reduced from 10s to 5s
        best_id = None
        attempts = 0
        max_attempts = 3  # Limit DOM search attempts
        
        logging.debug("[reply] Looking for reply ID, expected text: %r", expected_text[:50] if expected_text else None)
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            try:
                # Quick check - try to find any recent tweet with our text first
                cards = self.driver.find_elements(
                    By.XPATH, "//article[@data-testid='tweet' or @data-testid='post']"
                )[:5]  # Limit to first 5 cards to avoid hanging
                
                for i, card in enumerate(cards):
                    if time.time() > deadline:
                        logging.debug("[reply] Timeout during card %d examination", i)
                        break
                        
                    try:
                        # Quick text check with timeout
                        text_nodes = card.find_elements(By.CSS_SELECTOR, "div[data-testid='tweetText']")
                        if not text_nodes:
                            continue
                            
                        body = " ".join(n.text for n in text_nodes[:2]).strip()  # Limit to first 2 nodes
                        
                        # If we have expected text, verify it matches
                        if normalized_expected and len(normalized_expected) > 3:
                            if normalized_expected[:30] not in " ".join(body.split())[:100]:
                                continue
                        
                        # Try to extract ID from href
                        links = card.find_elements(By.XPATH, ".//a[contains(@href, '/status/')]")
                        for link in links[-2:]:  # Check last 2 links only
                            try:
                                href = link.get_attribute("href")
                                if href and "/status/" in href:
                                    potential_id = href.rstrip("/").split("/")[-1]
                                    if potential_id.isdigit() and len(potential_id) > 10:
                                        best_id = potential_id
                                        logging.debug("[reply] Found potential reply ID: %s", best_id)
                                        break
                            except Exception:
                                continue
                        
                        if best_id:
                            break
                            
                    except Exception as exc:
                        logging.debug("[reply] Card examination failed: %s", exc)
                        continue
                
                if best_id:
                    break
                    
                # Brief wait before retry, but not if we're near deadline
                if time.time() < deadline - 1:
                    time.sleep(0.5)
                    
            except Exception as exc:
                logging.debug("[reply] DOM search attempt %d failed: %s", attempts, exc)
                if time.time() < deadline - 0.5:
                    time.sleep(0.2)
        
        if best_id:
            logging.debug("[reply] ✅ Found reply ID: %s", best_id)
            return best_id
        
        logging.warning("[reply] ⚠️ No reply ID found after %d attempts", attempts)
        return None

        if self.username:
            cards = self.driver.find_elements(
                By.XPATH, "//article[@data-testid='tweet' or @data-testid='post' or @data-testid='cellInnerDiv']"
            )
            for card in cards[:10]:
                try:
                    author_link = card.find_element(By.CSS_SELECTOR, "div[data-testid='User-Name'] a[href*='x.com/']")
                    ahref = (author_link.get_attribute("href") or "").rstrip("/")
                    if ahref.split("/")[-1].lower() != (self.username or "").lower():
                        continue
                    link = card.find_element(By.XPATH, ".//a[contains(@href, '/status/')][last()]")
                    href = link.get_attribute("href")
                    if href:
                        return href.rstrip("/").split("/")[-1]
                except Exception:
                    continue

        with contextlib.suppress(Exception):
            with open("selenium_reply_id_debug.html", "w", encoding="utf-8") as handle:
                handle.write(self.driver.page_source)
        return best_id

    # --- fetch replies --------------------------------------------------

    def fetch_replies(self, conversation_id: str, since_id: Optional[str] = None) -> List[SocialPost]:
        url = f"{self.BASE_URL}/i/status/{conversation_id}"
        self.driver.get(url)
        self._wait_primary_column()
        replies: List[SocialPost] = []
        seen: set[str] = set()
        cutoff_val: Optional[int] = None
        if since_id is not None:
            with contextlib.suppress(ValueError):
                cutoff_val = int(str(since_id))

        deadline = time.time() + 15
        while time.time() < deadline:
            cards = self.driver.find_elements(
                By.XPATH, "//article[@data-testid='tweet' or @data-testid='post' or @data-testid='cellInnerDiv']"
            )
            for card in cards:
                post = self._extract_post(card, conversation_id)
                if not post or post.id == conversation_id:
                    continue
                if cutoff_val is not None:
                    with contextlib.suppress(ValueError):
                        if int(post.id) <= cutoff_val:
                            continue
                if post.id in seen:
                    continue
                seen.add(post.id)
                replies.append(post)
            if replies:
                break
            with contextlib.suppress(Exception):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.0)
        return replies


__all__ = ["SeleniumTwitterClient", "SocialPost"]
