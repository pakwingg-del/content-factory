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
    return (
        (config.get("indexnow_key") or "").strip()
        or (os.getenv(f"INDEXNOW_KEY_{site_id.upper()}") or "").strip()
        or (os.getenv("INDEXNOW_KEY") or "").strip()
    )

def fetch_recent_urls(config: dict, hours: int = 24, limit: int = 500):
    domain = (config.get("domain") or "").strip()
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
        urls = []
        for row in rows:
            slug = (row.get("url_slug") or "").strip().strip("/")
            if slug:
                urls.append(f"https://{domain}/{slug}/")

        print(f"📋 Found {len(urls)} URLs (last {hours}h) for {domain}")
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
    print(f"🚀 IndexNow → host={domain}, key={key[:8]}…, urls={len(urls)}")

    ok_batches = 0
    for i in range(0, len(urls), batch_size):
        chunk = urls[i:i + batch_size]
        payload = {
            "host": domain,
            "key": key,
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
    parser.add_argument("--limit", type=int, default=500, help="Max URLs (default 500)")
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat()}] IndexNow for site={args.site}")
    config = load_site_config(args.site)
    key = get_indexnow_key(config, args.site)
    domain, urls = fetch_recent_urls(config, hours=args.hours, limit=args.limit)
    submit_indexnow(domain, urls, key)

if __name__ == "__main__":
    main()
