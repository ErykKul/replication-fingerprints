#!/usr/bin/env python3
"""Test LibKey -> ProQuest -> PDF headless, capturing the PDF via network interception.

Loads the LibKey DOI page, follows the full-text-file link, and listens on ALL responses (across tabs) for
application/pdf -- the media.proquest.com PDF. If captured, the automation works for everything incl. APA.

Run:  python src/browser_libkey_test.py [doi]
"""
import sys, os
from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("data/.pw_profile")
DOI = sys.argv[1] if len(sys.argv) > 1 else "10.1037/0021-843x.115.4.798"
OUT = "data/libkey_test.pdf"


def main():
    captured = []
    if os.path.exists(OUT):
        os.remove(OUT)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=True, accept_downloads=True)

        def on_resp(resp):
            try:
                if "application/pdf" in resp.headers.get("content-type", ""):
                    b = resp.body()
                    if b[:5] == b"%PDF-":
                        captured.append(resp.url)
                        with open(OUT, "wb") as f:
                            f.write(b)
            except Exception:
                pass
        ctx.on("response", on_resp)

        page = ctx.new_page()
        page.goto(f"https://libkey.io/libraries/1781/{DOI}", wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(3000)
        ftf = [h for h in page.eval_on_selector_all("a", "els => els.map(e => e.href)") if "full-text-file" in h]
        print("full-text-file link:", ftf[:1])
        if ftf:
            try:
                page.goto(ftf[0], wait_until="networkidle", timeout=60000)
            except Exception as e:
                print("nav note:", str(e)[:50])
            page.wait_for_timeout(7000)
        ok = os.path.exists(OUT) and os.path.getsize(OUT) > 10000
        print("captured PDF urls:", [u[:70] for u in captured[:2]])
        print(f"RESULT: {'SUCCESS - ' + str(os.path.getsize(OUT)) + ' bytes' if ok else 'no PDF captured'}")
        ctx.close()


if __name__ == "__main__":
    main()
