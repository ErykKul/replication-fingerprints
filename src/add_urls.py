#!/usr/bin/env python3
"""Add a one-click KU Leuven EZproxy pull URL to the manifest + pull worklist.

KU Leuven off-campus access (per bib.kuleuven.be): https://kuleuven.e-bronnen.be/login?url=<target> logs you
into EZproxy (SSO) and resolves <target>. Using the DOI resolver as the target -> one link -> login ->
publisher full text -> download. (Eryk/LIBIS to confirm the exact pattern.)

Run:  python src/add_urls.py
"""
import os, glob
from urllib.parse import quote
import pandas as pd

PROXY = "https://kuleuven.e-bronnen.be/login?url="


def main():
    df = pd.read_csv("data/fulltext_manifest.csv")
    enc = df.doi.astype(str).map(lambda d: quote(d, safe="/"))   # URL-encode <>;:() etc. so DOIs resolve
    df["doi_url"] = "https://doi.org/" + enc
    df["ezproxy_url"] = PROXY + "https://doi.org/" + enc
    # LibKey (KU Leuven library id 1781): resolves any DOI through the library's holdings incl. APA -- best link
    df["libkey_url"] = "https://libkey.io/libraries/1781/" + enc
    n_special = (df.doi.astype(str) != enc).sum()
    print(f"URL-encoded {n_special} special-char DOIs; added LibKey links (work for APA via KU Leuven holdings)")
    # OA papers get a direct no-login PDF where we have it; else the ezproxy link
    df["best_url"] = df.apply(lambda r: r.oa_url if isinstance(r.oa_url, str) and r.oa_url.startswith("http")
                              else r.ezproxy_url, axis=1)
    df.to_csv("data/fulltext_manifest.csv", index=False)
    have = {os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md")}  # already got these
    pull = df[~df.paper_id.isin(have)][
        ["target_filename", "libkey_url", "ezproxy_url", "authors", "year", "title", "journal"]]
    pull.to_csv("data/fulltext_to_pull.csv", index=False)
    print(f"added ezproxy_url + best_url. pull worklist: {len(pull)} papers, each a one-click login+download link.")
    print("example ezproxy_url:", df.ezproxy_url.iloc[0])


if __name__ == "__main__":
    main()
