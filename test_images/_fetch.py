"""Download a few permissively-licensed hard-surface images from Wikimedia
Commons for testing the recognition pipeline. Run once:

    python test_images/_fetch.py

Picks clean-ish single-object shots (furniture / props / appliances) that stand
in for concept-art silhouettes. Each query keeps the first result whose pixels
fit a sane size; license is Commons (PD / CC) — see the page URL in _SOURCES.md.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "CubeGB-test-fetch/0.1 (recognition pipeline testing)"}

# (output stem, Commons search query) — hard-surface, clear silhouette, plain bg.
# Mix of box / cylinder / cone / sphere shapes so type recognition is exercised.
QUERIES = [
    ("chair", "chair"),
    ("traffic_cone", "traffic cone"),
    ("desk_lamp", "desk lamp"),
    ("fire_hydrant", "fire hydrant"),
    ("bucket", "bucket"),
    ("toaster", "toaster"),
]

API = "https://commons.wikimedia.org/w/api.php"


def _get(url: str, *, retries: int = 4, timeout: int = 60) -> bytes:
    """GET with polite throttle + exponential backoff on HTTP 429."""
    backoff = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    return json.loads(_get(url, timeout=30).decode("utf-8"))


def search_image(query: str) -> tuple[str, str] | None:
    """Return (image_url, page_url) for the first usable bitmap match."""
    data = _api({
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",      # File namespace
        "gsrlimit": "10",
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": "1024",     # ask for a 1024px-wide thumbnail
    })
    pages = (data.get("query") or {}).get("pages") or {}
    # Honour search rank (index) rather than dict order.
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 1e9))
    for p in ordered:
        info = (p.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        w, h = info.get("thumbwidth", 0), info.get("thumbheight", 0)
        # Skip extreme aspect ratios (banners, panoramas) — want a single object.
        if w and h and not (0.5 <= w / h <= 2.0):
            continue
        thumb = info.get("thumburl") or info.get("url")
        page = info.get("descriptionurl", "")
        if thumb:
            return thumb, page
    return None


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    sources = ["# Test image sources (Wikimedia Commons)\n"]
    for stem, query in QUERIES:
        try:
            hit = search_image(query)
        except Exception as exc:  # network hiccup — skip, keep going
            print(f"[skip] {stem}: {exc}")
            continue
        if not hit:
            print(f"[skip] {stem}: no match for {query!r}")
            continue
        img_url, page_url = hit
        ext = ".png" if img_url.lower().split("?")[0].endswith(".png") else ".jpg"
        out = HERE / f"{stem}{ext}"
        try:
            out.write_bytes(_get(img_url))
            print(f"[ok]   {out.name}  <-  {img_url}")
            sources.append(f"- **{out.name}** — {page_url}")
        except Exception as exc:
            print(f"[skip] {stem}: download failed: {exc}")
        time.sleep(1.5)  # be polite to the Commons API
    (HERE / "_SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
