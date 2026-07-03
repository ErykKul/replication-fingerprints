#!/usr/bin/env python3
"""Download full-text PDFs via LibKey (headless, IP-authenticated box).

Navigates each paper's libkey_direct link; LibKey routes to the publisher's direct PDF (ezproxy-rewritten,
e.g. onlinelibrary-wiley-com.kuleuven.e-bronnen.be/.../pdfdirect) or ProQuest media; we intercept the
application/pdf response (and download events) and save as pNNNN.pdf. Resumable, best-effort, polite.

Run:  python src/browser_libkey_download.py [max_this_run]
Then: python src/extract_pulled.py
"""
import os, sys, time, glob, pandas as pd
from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("data/.pw_profile")
PDF = "data/fulltext_pdf"


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    man = pd.read_csv("data/fulltext_manifest.csv")
    have = ({os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md")} |
            {os.path.basename(p)[:-4] for p in glob.glob(PDF + "/*.pdf")})
    ld = man.get("libkey_direct")
    todo = man[(~man.paper_id.isin(have)) & ld.astype(str).str.startswith("http")].head(cap)
    os.makedirs(PDF, exist_ok=True)
    print(f"attempting {len(todo)} via LibKey")
    cur = {"pid": None}
    got = 0
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=True, accept_downloads=True)

        def on_resp(resp):
            try:
                if cur["pid"] and "application/pdf" in resp.headers.get("content-type", ""):
                    b = resp.body()
                    if b[:5] == b"%PDF-" and len(b) > 10000:
                        with open(f"{PDF}/{cur['pid']}.pdf", "wb") as f:
                            f.write(b)
            except Exception:
                pass
        ctx.on("response", on_resp)

        def on_download(dl):
            try:
                if cur["pid"]:
                    dl.save_as(f"{PDF}/{cur['pid']}.pdf")
            except Exception:
                pass
        page = ctx.new_page()
        page.on("download", on_download)
        page.set_default_timeout(20000)
        for _, r in todo.iterrows():
            out = f"{PDF}/{r.paper_id}.pdf"
            if os.path.exists(out):
                continue
            cur["pid"] = r.paper_id
            try:
                page.goto(r.libkey_direct, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(4000)
            except Exception:
                pass
            ok = os.path.exists(out) and os.path.getsize(out) > 10000
            got += 1 if ok else 0
            print(f"  {'OK ' if ok else '-- '} {r.paper_id}  {str(r.title)[:44]}")
            time.sleep(1)
        ctx.close()
    print(f"\ndownloaded {got}. Re-run to continue; then src/extract_pulled.py.")


if __name__ == "__main__":
    main()
