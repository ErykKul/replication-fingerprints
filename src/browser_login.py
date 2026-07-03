#!/usr/bin/env python3
"""One-time KU Leuven login for the download automation (saves a persistent browser session).

Opens a real Chromium window with a persistent profile so you do SSO + 2FA ONCE; the session is saved in
data/.pw_profile/ and reused by browser_download.py (no re-login until it expires). If KUL_USER / KUL_PASS
are set in paper3/.env they're typed for you (2FA still on your phone). Needs a display (headed window).

Setup (once):
  .venv/bin/pip install playwright python-dotenv
  .venv/bin/playwright install chromium
Run:
  python src/browser_login.py
"""
import os
from playwright.sync_api import sync_playwright
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

PROFILE = os.path.abspath("data/.pw_profile")
# any DOI triggers the EZproxy login flow; land on the publisher article page to confirm success
TRIGGER = "https://kuleuven.e-bronnen.be/login?url=https://doi.org/10.1037/a0029288"


def main():
    os.makedirs(PROFILE, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=False, accept_downloads=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(TRIGGER, wait_until="domcontentloaded")
        u, pw = os.getenv("KUL_USER"), os.getenv("KUL_PASS")
        if u and pw:
            for sel in ('input[name="username"]', '#username', 'input[type="email"]', 'input[name="j_username"]'):
                if page.query_selector(sel):
                    page.fill(sel, u); break
            for sel in ('input[name="password"]', '#password', 'input[type="password"]', 'input[name="j_password"]'):
                if page.query_selector(sel):
                    page.fill(sel, pw); break
            print("Pre-filled username/password from .env; approve 2FA on your phone, then continue in the window.")
        input("Finish logging in (SSO + 2FA) until you SEE a publisher article page, then press Enter here... ")
        ctx.close()
        print("Session saved to", PROFILE, "- now run: src/browser_download.py")


if __name__ == "__main__":
    main()
