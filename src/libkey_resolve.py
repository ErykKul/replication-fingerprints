#!/usr/bin/env python3
"""Resolve every pull-list DOI to its LibKey direct full-text link via the ThirdIron API.

Requests can't get the anonymous LibKey token (session handshake), so we grab it once with Playwright, then
call the article API per DOI through the browser's request context. Writes `libkey_direct` (the working
/articles/<id>/full-text-file link -- the format that works for Eryk) into the manifest + pull list, and
flags retractions and papers not in KU Leuven holdings.

Run:  python src/libkey_resolve.py
"""
import os, time, glob
from urllib.parse import quote
import pandas as pd
from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("data/.pw_profile")
LIB = "1781"


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")
    have = {os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md")}
    todo = man[~man.paper_id.isin(have)].reset_index(drop=True)
    print(f"resolving {len(todo)} DOIs via LibKey...")
    rows = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=True)
        tok = {}
        ctx.on("response", lambda r: tok.__setitem__("t", r.request.headers.get("authorization"))
               if (r.request.headers.get("authorization") and "api.thirdiron.com" in r.url) else None)
        page = ctx.new_page()
        page.goto(f"https://libkey.io/libraries/{LIB}/10.1037/0021-843x.115.4.798", wait_until="networkidle", timeout=40000)
        page.wait_for_timeout(5000)
        t = tok.get("t")
        if not t:
            print("could not obtain LibKey token"); ctx.close(); return
        h = {"Authorization": t, "Origin": "https://libkey.io", "Referer": "https://libkey.io/"}
        nlink = nretr = 0
        for i, r in todo.iterrows():
            link, hold, retr = "", None, False
            try:
                resp = ctx.request.get(f"https://api.thirdiron.com/v2/articles/{quote('doi:' + str(r.doi), safe='')}",
                                       headers=h, timeout=20000)
                if resp.ok:
                    a = resp.json().get("data", {}).get("attributes", {})
                    link = a.get("libkeyFullTextFile") or ""
                    hold = a.get("withinLibraryHoldings")
                    retr = bool(a.get("retractionDoi"))
            except Exception:
                pass
            rows.append({"paper_id": r.paper_id, "libkey_direct": link, "in_holdings": hold, "retracted": retr})
            nlink += bool(link); nretr += retr
            if i % 50 == 0:
                print(f"  ...{i}/{len(todo)} ({nlink} full-text links so far)")
            time.sleep(0.12)
        ctx.close()
    res = pd.DataFrame(rows)
    res.to_csv("data/libkey_resolved.csv", index=False)
    # merge into manifest + regenerate pull list with libkey_direct as the primary link
    man = man.merge(res, on="paper_id", how="left")
    man.to_csv("data/fulltext_manifest.csv", index=False)
    pull = man[~man.paper_id.isin(have)][
        ["target_filename", "libkey_direct", "ezproxy_url", "retracted", "authors", "year", "title", "journal"]]
    pull.to_csv("data/fulltext_to_pull.csv", index=False)
    print(f"\nresolved {len(res)} | LibKey full-text link found: {nlink} ({100*nlink/len(res):.0f}%) | "
          f"retractions flagged: {nretr}")
    print("wrote libkey_direct into fulltext_to_pull.csv (primary link) + data/libkey_resolved.csv")


if __name__ == "__main__":
    main()
