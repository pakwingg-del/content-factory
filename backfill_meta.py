#!/usr/bin/env python3
"""One-shot: rebuild short meta_description rows from title, then ready for IndexNow.

Usage:
  python backfill_meta.py --site viralnn
  python backfill_meta.py --site all
  python backfill_meta.py --site viralnn --dry-run
  python backfill_meta.py --site viralnn --submit-indexnow

Selects rows where meta is NULL/empty or length(meta_description) < 120,
rebuilds with the same fit_meta_description logic as generator.py (150–160),
UPDATEs D1, writes sites/<site>/last_run_urls.json for --from-last-run IndexNow.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Import fit_meta from generator if available; else local copy.
try:
    from generator import fit_meta_description, to_canonical_url  # type: ignore
except Exception:
    fit_meta_description = None  # patched below


def load_site_config(site_id: str) -> dict:
    path = Path(f"sites/{site_id}/config.json")
    if not path.exists():
        print(f"❌ Config not found: {path}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def env_from_config(config: dict, key: str, fallback_env: str = None):
    env_name = config.get(key) or fallback_env
    if not env_name:
        return None
    return os.getenv(env_name)


def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("https://", "").replace("http://", "").strip("/")
    if d.startswith("www."):
        d = d[4:]
    return d


def canonical_url(domain: str, slug: str) -> str:
    slug = (slug or "").strip().strip("/")
    if not slug:
        return f"https://{domain}/"
    return f"https://{domain}/{slug}/"


def local_fit_meta(text, title="", keyword="", min_len=150, max_len=160):
    banned = re.compile(r"here's what is unfolding", re.IGNORECASE)
    text = (text or "").strip()
    text = re.sub(r'^["\']|["\']$', "", text).strip()
    text = banned.sub("", text)
    text = " ".join(text.split())
    base = title or keyword or "Latest updates"
    base = banned.sub("", base).strip()
    if not text or banned.search(text) or len(text) < 40:
        if keyword:
            text = (
                f"{base.rstrip('.')} — who it hits, what it costs, and the money-math "
                f"US readers care about on {keyword}."
            )
        else:
            text = (
                f"{base.rstrip('.')} — who it hits, what changes next, and why it "
                f"matters for American readers right now."
            )
        text = " ".join(text.split())
    if len(text) < min_len:
        soft = (
            f" Quick read on who is affected and what {keyword} means for wallets."
            if keyword
            else " Quick read on who is affected and what changes next."
        )
        if soft.strip().lower() not in text.lower():
            text = (text.rstrip(". ") + "." + soft).strip()
        text = " ".join(text.split())
    text = banned.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        cut = text[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.rstrip(".,;:") + "…"
    while len(text) < min_len:
        text = (text.rstrip(". ") + ". Updated for American readers.").strip()
        text = " ".join(text.split())
        if len(text) > max_len:
            cut = text[: max_len - 1]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            text = cut.rstrip(".,;:") + "…"
            break
    return text


fit_meta = fit_meta_description or local_fit_meta


def d1_query(account_id, database_id, api_token, sql, params=None):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {"sql": sql, "params": params or []}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    data = resp.json()
    if resp.status_code != 200 or not data.get("success"):
        raise RuntimeError(f"D1 query failed: {resp.text[:400]}")
    return data.get("result", [{}])[0].get("results", [])


def d1_batch(account_id, database_id, api_token, statements):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json={"batch": statements}, timeout=120)
    data = resp.json()
    if resp.status_code != 200 or not data.get("success"):
        raise RuntimeError(f"D1 batch failed: {resp.text[:400]}")
    return data


def backfill_site(site_id: str, min_len: int, dry_run: bool, limit: int) -> list:
    config = load_site_config(site_id)
    domain = normalize_domain(config.get("domain") or "")
    account_id = env_from_config(config, "cloudflare_account_id_env", "CLOUDFLARE_ACCOUNT_ID")
    database_id = env_from_config(config, "cloudflare_database_id_env", "CLOUDFLARE_DATABASE_ID")
    api_token = env_from_config(config, "cloudflare_api_token_env", "CLOUDFLARE_API_TOKEN")
    if not all([domain, account_id, database_id, api_token]):
        print(f"❌ [{site_id}] missing domain or Cloudflare credentials")
        sys.exit(1)

    # SQLite length() counts characters for TEXT
    sql = """
        SELECT url_slug, title, keyword, meta_description
        FROM articles
        WHERE meta_description IS NULL
           OR TRIM(meta_description) = ''
           OR length(meta_description) < ?
        ORDER BY created_at DESC
        LIMIT ?
    """
    rows = d1_query(account_id, database_id, api_token, sql, [min_len, limit])
    print(f"[{site_id}] short/empty meta rows: {len(rows)} (min_len={min_len}, limit={limit})")
    if not rows:
        return []

    updates = []
    urls = [f"https://{domain}/"]
    seen = {urls[0]}
    for row in rows:
        title = (row.get("title") or "").strip()
        keyword = (row.get("keyword") or "").strip()
        old = (row.get("meta_description") or "").strip()
        new = fit_meta(old, title=title, keyword=keyword, min_len=150, max_len=160)
        slug = row.get("url_slug") or ""
        u = canonical_url(domain, slug)
        if u not in seen:
            seen.add(u)
            urls.append(u)
        updates.append(
            {
                "sql": "UPDATE articles SET meta_description = ? WHERE url_slug = ?",
                "params": [new, slug],
            }
        )
        if len(updates) <= 3:
            print(f"  sample {slug}: {len(old)} → {len(new)} | {new[:90]}…")

    if dry_run:
        print(f"[{site_id}] dry-run: would UPDATE {len(updates)} rows, IndexNow {len(urls)} URLs")
        return urls

    chunk = 40
    for i in range(0, len(updates), chunk):
        d1_batch(account_id, database_id, api_token, updates[i : i + chunk])
        print(f"  ✅ updated chunk {i // chunk + 1} ({min(i + chunk, len(updates))}/{len(updates)})")
        time.sleep(0.2)

    out = Path(f"sites/{site_id}/last_run_urls.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "site_id": site_id,
        "domain": domain,
        "generated_at": datetime.now().isoformat(),
        "article_count": max(0, len(urls) - 1),
        "urls": urls,
        "note": "meta backfill",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{site_id}] wrote {out} ({len(urls)} URLs)")
    return urls


def main():
    parser = argparse.ArgumentParser(description="Backfill short meta_description in D1")
    parser.add_argument("--site", required=True, help="site id or 'all'")
    parser.add_argument("--min-len", type=int, default=120, help="Treat meta shorter than this as short (default 120)")
    parser.add_argument("--limit", type=int, default=5000, help="Max rows per site (default 5000)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-indexnow", action="store_true", help="After update, run submit_indexing --from-last-run")
    args = parser.parse_args()

    sites = (
        ["viralnn", "billcutdaily", "gadgetpulseus", "popspilldaily"]
        if args.site == "all"
        else [args.site]
    )
    for site in sites:
        backfill_site(site, min_len=args.min_len, dry_run=args.dry_run, limit=args.limit)
        if args.submit_indexnow and not args.dry_run:
            import subprocess

            print(f"[{site}] IndexNow --from-last-run …")
            subprocess.check_call([sys.executable, "submit_indexing.py", "--site", site, "--from-last-run"])


if __name__ == "__main__":
    main()
