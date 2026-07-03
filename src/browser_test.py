#!/usr/bin/env python3
"""Non-interactive login test: does the EZproxy pattern resolve, and do we have a display?

Launches Chromium (headed first, then headless if no display), navigates to a sample EZproxy link,
screenshots what it lands on (data/login_test.png), and prints the final URL + title so we can confirm the
pattern reaches the KU Leuven SSO login.

Run:  python src/browser_test.py
"""
from playwright.sync_api import sync_playwright

URL = "https://kuleuven.e-bronnen.be/login?url=https://doi.org/10.1037/a0029288"


def try_launch(headless):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(URL, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(3500)
        page.screenshot(path="data/login_test.png")
        print("MODE:", "headless" if headless else "headed")
        print("final_url:", page.url)
        print("title:", page.title())
        browser.close()


def main():
    try:
        try_launch(headless=False)
        print("DISPLAY_AVAILABLE=yes")
    except Exception as e:
        print("headed failed:", str(e)[:90], "-> headless")
        try_launch(headless=True)
        print("DISPLAY_AVAILABLE=no")


if __name__ == "__main__":
    main()
