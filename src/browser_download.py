#!/usr/bin/env python3
"""Auto-download the paywalled PDFs via the saved KU Leuven session (run browser_login.py first).

For each paper still missing locally: open its EZproxy URL (already logged in), find the PDF via the generic
`citation_pdf_url` meta tag (set by most publishers) or a .pdf link, and download it to
data/fulltext_pdf/pNNNN.pdf using the authenticated session. Resumable, polite (1.5s/paper). Best-effort:
publishers that hide the PDF behind a viewer/captcha are skipped and stay on the manual pull list.

Run:  python src/browser_download.py [max_this_run]
Then: python src/extract_pulled.py   (PDF -> MD)
"""
import os, sys, time, glob, pandas as pd
from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("data/.pw_profile")
PDF = "data/fulltext_pdf"


def save_pdf(ctx, url, out):
    try:
        r = ctx.request.get(url, timeout=30000)
        b = r.body()
        if r.ok and b[:5] == b"%PDF-":
            with open(out, "wb") as f:
                f.write(b)
            return True
    except Exception:
        pass
    return False


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    df = pd.read_csv("data/fulltext_manifest.csv")
    have = ({os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md")} |
            {os.path.basename(p)[:-4] for p in glob.glob(PDF + "/*.pdf")})
    todo = df[~df.paper_id.isin(have)].head(cap)
    os.makedirs(PDF, exist_ok=True)
    print(f"attempting {len(todo)} papers (session: {PROFILE})")
    got = fail = 0
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=True, accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(20000)
        for _, r in todo.iterrows():
            out = f"{PDF}/{r.target_filename}"
            ok = False
            try:
                page.goto(r.ezproxy_url, timeout=25000, wait_until="domcontentloaded")
                url = page.get_attribute('meta[name="citation_pdf_url"]', "content")
                if url:
                    ok = save_pdf(ctx, url, out)
                if not ok:
                    a = page.query_selector('a[href$=".pdf"], a[href*=".pdf?"]')
                    href = a.get_attribute("href") if a else None
                    if href:
                        if href.startswith("/"):
                            base = "/".join(page.url.split("/")[:3])
                            href = base + href
                        ok = save_pdf(ctx, href, out)
            except Exception as e:
                print(f"  ERR {r.target_filename} {str(e)[:40]}")
            got += 1 if ok else 0
            fail += 0 if ok else 1
            print(f"  {'OK ' if ok else '-- '} {r.target_filename}  {str(r.title)[:46]}")
            time.sleep(1.5)
        ctx.close()
    print(f"\ndownloaded {got}, failed {fail}. Re-run to continue; then run src/extract_pulled.py.")


if __name__ == "__main__":
    main()
