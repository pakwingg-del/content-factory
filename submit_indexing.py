import json
import os
import sys
import time
import argparse
import requests
from datetime import datetime


def load_site_config(site_id: str) -> dict:
    config_path = f"sites/{site_id}/config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def env_from_config(config: dict, key: str, fallback_env: str = None):
    env_name = config.get(key) or fallback_env
    if not env_name:
        return None
    return os.getenv(env_name)


def get_indexnow_key(config: dict, site_id: str) -> str:
    """Resolve IndexNow key from env. Never treat env var NAMES as the key value."""
    candidates = []

    env_name = (config.get("indexnow_key_env") or "").strip()
    if env_name:
        candidates.append(os.getenv(env_name))

    raw = (config.get("indexnow_key") or "").strip()
    if raw:
        looks_like_env_name = raw.startswith("INDEXNOW_") or raw == "INDEXNOW_KEY"
        if looks_like_env_name:
            candidates.append(os.getenv(raw))
        else:
            candidates.append(raw)

    candidates.append(os.getenv(f"INDEXNOW_KEY_{site_id.upper()}"))
    candidates.append(os.getenv("INDEXNOW_KEY"))

    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    return ""


def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("https://", "").replace("http://", "").strip("/")
    if d.startswith("www."):
        d = d[4:]
    return d


def to_canonical_url(domain: str, slug: str) -> str:
    slug = (slug or "").strip().strip("/")
    if not slug:
        return f"https://{domain}/"
    return f"https://{domain}/{slug}/"


def fetch_recent_urls(config: dict, hours: int = 24, limit: int = 500):
    domain = normalize_domain(config.get("domain") or "")
    if not domain:
        print("❌ domain missing in config.json")
        sys.exit(1)

    account_id = env_from_config(config, "cloudflare_account_id_env", "CLOUDFLARE_ACCOUNT_ID")
    database_id = env_from_config(config, "cloudflare_database_id_env", "CLOUDFLARE_DATABASE_ID")
    api_token = env_from_config(config, "cloudflare_api_token_env", "CLOUDFLARE_API_TOKEN")
    if not all([account_id, database_id, api_token]):
        print("❌ Missing Cloudflare credentials (check config + GitHub Secrets)")
        sys.exit(1)

    since = int(time.time()) - hours * 3600
    sql = """
        SELECT url_slug
        FROM articles
        WHERE created_at >= ?
        ORDER BY created_at DESC
        LIMIT ?
    """
    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {"sql": sql, "params": [since, limit]}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code != 200 or not data.get("success"):
            print(f"❌ D1 query failed: {resp.text[:300]}")
            return domain, []

        rows = data.get("result", [{}])[0].get("results", [])
        urls = [f"https://{domain}/"]
        seen = {urls[0]}
        for row in rows:
            u = to_canonical_url(domain, row.get("url_slug") or "")
            if u not in seen:
                seen.add(u)
                urls.append(u)

        print(f"📋 Found {len(urls)} URLs (incl. homepage, last {hours}h) for {domain}")
        if urls[1:2]:
            print(f"   sample: {urls[1]}")
        return domain, urls
    except Exception as e:
        print(f"❌ D1 error: {e}")
        return domain, []


def submit_indexnow(domain: str, urls: list, key: str):
    if not urls:
        print("⚠️ No URLs to submit")
        return
    if not key:
        print("❌ INDEXNOW_KEY missing")
        sys.exit(1)

    endpoint = "https://api.indexnow.org/indexnow"
    batch_size = 200
    key_location = f"https://{domain}/{key}.txt"
    print(f"🚀 IndexNow → host={domain}")
    print(f"   keyLocation={key_location}")
    print(f"   urls={len(urls)}")

    ok_batches = 0
    for i in range(0, len(urls), batch_size):
        chunk = urls[i:i + batch_size]
        payload = {
            "host": domain,
            "key": key,
            "keyLocation": key_location,
            "urlList": chunk
        }
        try:
            r = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if r.status_code in (200, 202):
                ok_batches += 1
                print(f"  ✅ Batch {i // batch_size + 1}: {len(chunk)} URLs → {r.status_code}")
            else:
                print(f"  ❌ Batch {i // batch_size + 1}: {r.status_code} — {r.text[:200]}")
        except Exception as e:
            print(f"  ❌ Batch error: {e}")
        time.sleep(0.4)

    print(f"🎉 IndexNow finished — ok batches: {ok_batches}")


def main():
    parser = argparse.ArgumentParser(description="Submit recent article URLs to IndexNow (Bing etc.)")
    parser.add_argument("--site", required=True, help="Site id, e.g. viralnn / popspilldaily")
    parser.add_argument("--hours", type=int, default=24, help="Look back hours (default 24)")
    parser.add_argument("--limit", type=int, default=500, help="Max article URLs (default 500)")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Extra canonical URL to include (repeatable)"
    )
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat()}] IndexNow for site={args.site}")
    config = load_site_config(args.site)
    key = get_indexnow_key(config, args.site)
    domain, urls = fetch_recent_urls(config, hours=args.hours, limit=args.limit)

    extra = []
    for raw in args.url:
        raw = (raw or "").strip()
        if not raw:
            continue
        if raw.startswith("http"):
            u = raw if raw.endswith("/") else raw + "/"
        else:
            u = to_canonical_url(domain, raw)
        extra.append(u)

    merged = []
    seen = set()
    for u in extra + urls:
        if u not in seen:
            seen.add(u)
            merged.append(u)

    submit_indexnow(domain, merged, key)


if __name__ == "__main__":
    main()
