import json
import os
import time
import requests
import sys
import re
import argparse
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================== 配置 ======================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
LLM_EXTRA_BODY = {
    "thinking": {"type": "disabled"},
    "reasoning_effort": "none",
}


DEFAULT_PERSONA_MATRIX = [
    "Tabloid journalist, use heavy dramatic language, ALL CAPS hooks, shocking reveals, urgent tone, and American sensational style.",
    "Viral Gen-Z TikToker, high energy brainrot slang, short punchy sentences, heavy hype, emojis vibe, and trending American internet culture.",
    "Cynical Reddit user, heavy sarcasm, dark humor, AITA style commentary, and US-centric internet slang.",
    "Deep conspiracy investigator, 'hidden truth', 'stay woke', connecting dots others miss, with American political and cultural angle.",
    "Moral critic and societal observer, focus on ethical issues, 'society is collapsing' angle, and impact on American daily life."
]
FINANCIAL_PERSONA_MATRIX = [
    "Sharp financial analyst, clear data-driven language, focus on market impact and investor implications.",
    "Crypto Twitter trader voice, urgent, slang-heavy, FOMO and risk warnings mixed.",
    "Skeptical market observer, question hype, point out risks and who benefits."
]


def load_site_config(site_id: str) -> dict:
    config_path = f"sites/{site_id}/config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        print("   Please create sites/{site_id}/config.json")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"✅ Loaded config for site: {config.get('site_id')} ({config.get('domain')})")
    return config


def env_from_config(config: dict, key: str, fallback_env: str = None):
    env_name = config.get(key) or fallback_env
    if not env_name:
        return None
    return os.getenv(env_name)


def get_persona_matrix(config: dict):
    style = (config.get("persona_style") or "default").lower()
    if style == "financial":
        return FINANCIAL_PERSONA_MATRIX
    return DEFAULT_PERSONA_MATRIX


def download_image(url, filename):
    try:
        os.makedirs("public/images", exist_ok=True)
        filepath = f"public/images/{filename}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return f"/images/{filename}"
    except Exception:
        return None
    return None


def get_pexels_image(query):
    if not PEXELS_API_KEY:
        return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": query, "per_page": 1, "orientation": "landscape"}
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("photos"):
                photo = data["photos"][0]
                image_url = photo["src"]["large"]
                filename = f"{int(time.time())}_{photo['id']}.jpg"
                return download_image(image_url, filename)
    except Exception:
        pass
    return None


def fit_title(text, keyword="", min_len=50, max_len=65):
    """強制 title 50–65 字元（Bing 警告 >70）。"""
    t = (text or "").strip()
    t = re.sub(r'^["\']|["\']$', "", t).strip()
    t = re.sub(
        r"^(FOR IMMEDIATE RELEASE|BREAKING NEWS|BREAKING|HEADLINE|TITLE|UPDATE)[:\s]*",
        "",
        t,
        flags=re.IGNORECASE
    ).strip()
    t = " ".join(t.split())

    if len(t) < min_len:
        extra = f" — {keyword} update" if keyword else " — latest US update"
        if extra.lower() not in t.lower():
            t = (t + extra).strip()
        t = " ".join(t.split())

    if len(t) > max_len:
        cut = t[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        t = cut.rstrip(" ,.;:|-—") + "…"

    # Keep padded title; never collapse back to bare keyword
    if len(t) < 40 and keyword:
        extra = f" — {keyword} update"
        if extra.lower() not in t.lower():
            t = (t + extra).strip()
        t = " ".join(t.split())
        if len(t) > max_len:
            cut = t[: max_len - 1]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            t = cut.rstrip(" ,.;:|-—") + "…"

    return t


def fit_meta_description(text, title="", keyword="", min_len=150, max_len=160):
    """強制 meta description 落喺 150–160 字元（Bing 建議）。"""
    text = (text or "").strip()
    text = re.sub(r'^["\']|["\']$', "", text).strip()
    text = " ".join(text.split())
    fillers = [
        f" Here's what is unfolding around {keyword} and why US readers are paying attention right now.",
        f" See why {keyword} is trending and what it could mean for people across America today.",
        f" A clear breakdown of {keyword}, the key claims, and why this story is gaining traction now.",
    ]
    guard = 0
    while len(text) < min_len and guard < 5:
        if not text:
            text = title or f"Latest updates on {keyword}"
        filler = fillers[guard % len(fillers)]
        if filler.strip() not in text:
            text = (text.rstrip(". ") + "." + filler).strip()
        else:
            text = (text + " Stay informed with the latest verified developments.").strip()
        text = " ".join(text.split())
        guard += 1
    if len(text) > max_len:
        cut = text[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.rstrip(".,;:") + "…"
    if len(text) < min_len:
        pad = " More details inside."
        text = (text + pad)[:max_len]
    return text



def extract_text(completion) -> str:
    """DeepSeek flash may put body in reasoning_* when thinking is on; prefer content."""
    msg = completion.choices[0].message
    text = (getattr(msg, "content", None) or "") or ""
    if not str(text).strip():
        text = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
    if not str(text).strip() and getattr(msg, "model_extra", None):
        extra = msg.model_extra or {}
        text = extra.get("reasoning_content") or extra.get("reasoning") or ""
    return (text or "").strip()


def fetch_single_article(persona_tuple, seed, last_updated, site_config):
    round_idx, current_persona = persona_tuple
    query = seed["query"]
    system_prompt = (
        f"You are a: {current_persona}. Write a unique, engaging viral news article for an American audience.\n"
        f"CRITICAL RULES:\n"
        f"- The VERY FIRST LINE must be the title only (no labels like TITLE: or BREAKING:).\n"
        f"- Title MUST be 50-65 characters including spaces. Count carefully. Never exceed 65.\n"
        f"- Curiosity-driven, specific, natural. No ALL CAPS. No repeating the same formula.\n"
        f"- Write the main article body (800-1100 words).\n"
        f"- After the body, add a short closing opinion (2-3 sentences).\n"
        f"- Use American English."
    )
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a viral article about: {query}"}
            ],
            max_tokens=1600,
            temperature=0.85,
            extra_body=LLM_EXTRA_BODY,
        )
        content = extract_text(completion)
        finish = getattr(completion.choices[0], "finish_reason", None)
        print(
            f"model reply chars: {len(content)} finish: {finish} first80: {content[:80]!r}"
        )
        if len(content) < 100:
            print(f"⚠️ thin/empty model reply for {query!r} — skip")
            return None

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        raw_title = lines[0] if lines else query
        clean_title = fit_title(raw_title, keyword=query, min_len=50, max_len=65)

        body_only = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
        if len(re.sub(r"\s+", " ", body_only)) < 200:
            body_only = content

        # One API call: meta from local fitter (no second LLM call)
        meta_description = fit_meta_description(
            "",
            title=clean_title,
            keyword=query,
            min_len=150,
            max_len=160,
        )

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body_only) if p.strip()]
        if "final thoughts" in body_only.lower():
            final_content = body_only
        elif len(paragraphs) >= 2:
            closer = re.sub(r"<[^>]+>", "", paragraphs[-1]).strip()
            main = "\n\n".join(paragraphs[:-1])
            final_content = (
                main + "\n\n<h3>Final Thoughts</h3>\n<p>" + closer[:500] + "</p>"
            )
        else:
            final_content = body_only

        image_path = get_pexels_image(query)
        if image_path:
            final_content = (
                f'<img src="{image_path}" alt="{clean_title}" '
                f'style="max-width:100%; height:auto; border-radius:8px;"><br><br>'
                + final_content
            )

        body_text = re.sub(r"<[^>]+>", " ", final_content or "")
        body_text = re.sub(r"\s+", " ", body_text).strip()
        clean_title = fit_title(clean_title, keyword=query)
        if len(clean_title) < 40 or len(body_text) < 400:
            print(
                f"⚠️ drop thin article: {query!r} title={len(clean_title)} body={len(body_text)}"
            )
            return None

        return {
            "keyword": query,
            "persona_id": round_idx + 1,
            "persona_type": current_persona.split(",")[0],
            "title": clean_title,
            "meta_description": meta_description,
            "body": final_content,
            "source_volume": seed.get("search_volume", 0),
            "generated_at": last_updated
        }
    except Exception as e:
        print(f"⚠️ Error processing {query}: {e}")
        return None



def generate_sitemap():
    print("🗺️ Generating sitemap...")
    try:
        print("✅ Sitemap generated (placeholder)")
    except Exception:
        pass


def generate_matrix(config: dict):
    domain = config.get("domain", "example.com")
    trends_url = config.get(
        "trends_url",
        "https://raw.githubusercontent.com/pakwingg-del/Trends-Hub/main/master_trends.json"
    )
    trends_limit = int(config.get("trends_limit", 80))
    personas = get_persona_matrix(config)
    personas_count = min(int(config.get("personas_count", len(personas))), len(personas))
    personas = personas[:personas_count]
    target_articles = trends_limit * personas_count
    print(f"📡 Fetching trends for {domain}...")
    print(f"   Target ~{target_articles} articles ({trends_limit} trends × {personas_count} personas)")
    print(f"   LLM model: {LLM_MODEL}")

    try:
        response = requests.get(trends_url, timeout=20)
        response.raise_for_status()
        data = response.json()
        trending_seeds = data.get("trending_seeds", [])
        trending_seeds.sort(
            key=lambda x: (x.get("increase", 0), x.get("search_volume", 0)),
            reverse=True
        )
        # Niche filters from site config (include OR; exclude any match)
        kw_filter = [k.lower() for k in (config.get("keyword_filter") or []) if k]
        kw_exclude = [k.lower() for k in (config.get("keyword_exclude") or []) if k]

        def _seed_ok(seed):
            q = (seed.get("query") or "").lower()
            if not q:
                return False
            if kw_exclude and any(x in q for x in kw_exclude):
                return False
            if kw_filter and not any(x in q for x in kw_filter):
                return False
            return True

        if kw_filter or kw_exclude:
            before = len(trending_seeds)
            trending_seeds = [s for s in trending_seeds if _seed_ok(s)]
            print(
                f"🧹 Filter: {before} → {len(trending_seeds)} "
                f"(filter={len(kw_filter)} exclude={len(kw_exclude)})"
            )
            if len(trending_seeds) < 5:
                print("❌ Too few trends after keyword_filter/exclude — check config + Trends JSON")
                sys.exit(1)

        seeds = trending_seeds[:trends_limit]
        print(f"✅ Loaded Top {len(seeds)} trends")
    except Exception as e:
        print(f"❌ Error fetching trends: {e}")
        sys.exit(1)

    all_articles = []
    MAX_WORKERS = int(config.get("max_workers", 8))
    print(f"🚀 Starting generation for [{config.get('site_id')}]...")
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for round_idx, persona in enumerate(personas):
            for seed in seeds:
                tasks.append(
                    executor.submit(
                        fetch_single_article,
                        (round_idx, persona),
                        seed,
                        datetime.now().isoformat(),
                        config
                    )
                )
        completed_count = 0
        for future in as_completed(tasks):
            result = future.result()
            if result:
                all_articles.append(result)
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(tasks):
                print(f"📦 Progress: {completed_count}/{len(tasks)}")

    attempted = len(tasks)
    kept = len(all_articles)
    print(f"✅ Generated {kept}/{attempted} usable articles")
    min_keep = max(5, int(attempted * 0.3))
    if kept < min_keep:
        print(f"❌ Too few usable articles ({kept} < {min_keep}) — abort before D1 inject")
        sys.exit(1)
    for sample in all_articles[:3]:
        t = sample.get("title", "")
        md = sample.get("meta_description", "")
        print(f"📰 Title sample ({len(t)} chars): {t}")
        print(f"📝 Meta sample ({len(md)} chars): {md}")

    account_id = env_from_config(config, "cloudflare_account_id_env", "CLOUDFLARE_ACCOUNT_ID")
    database_id = env_from_config(config, "cloudflare_database_id_env", "CLOUDFLARE_DATABASE_ID")
    api_token = env_from_config(config, "cloudflare_api_token_env", "CLOUDFLARE_API_TOKEN")
    if not all([account_id, database_id, api_token]):
        print("❌ Missing Cloudflare credentials (check config + GitHub Secrets)")
        sys.exit(1)

    print(f"🚀 Injecting {len(all_articles)} articles into D1 ({domain})...")
    current_time = int(time.time())
    now = datetime.now()
    year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
    statements = []
    for idx, article in enumerate(all_articles):
        safe_keyword = "".join([c if c.isalnum() else "_" for c in article["keyword"]]).lower()
        url_slug = f"{year}/{month}/{day}/{safe_keyword}_{idx}"
        article_body = article["body"]
        ad_str = config.get("ad_verification")
        if ad_str and idx == 0:
            article_body += f"\n\nAdsterra verification string: {ad_str}"
        title = fit_title(article.get("title", ""), keyword=article.get("keyword", ""))
        meta = fit_meta_description(
            article.get("meta_description", ""),
            title=title,
            keyword=article.get("keyword", "")
        )
        sql = """INSERT OR REPLACE INTO articles
                 (title, keyword, body, persona_id, persona_type, search_volume, created_at, url_slug, meta_description)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);"""
        params = [
            title,
            article["keyword"],
            article_body,
            article["persona_id"],
            article["persona_type"],
            str(article["source_volume"]),
            current_time,
            url_slug,
            meta
        ]
        statements.append({"sql": sql, "params": params})

    chunk_size = 50
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    has_error = False
    for i in range(0, len(statements), chunk_size):
        chunk = statements[i:i + chunk_size]
        payload = {"batch": chunk}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200 and response.json().get("success"):
                print(f"✅ Injected chunk {i // chunk_size + 1}")
            else:
                print(f"❌ Failed chunk {i // chunk_size + 1}: {response.text}")
                has_error = True
        except Exception as e:
            print(f"⚠️ Error in chunk {i // chunk_size + 1}: {e}")
            has_error = True

    if has_error:
        print("❌ MISSION FAILED")
        sys.exit(1)
    print("🎉 All articles injected into D1!")
    generate_sitemap()
    print(f"🎉 [{config.get('site_id')}] Batch Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-site content generator")
    parser.add_argument("--site", required=True, help="Site id, e.g. viralnn")
    args = parser.parse_args()
    config = load_site_config(args.site)
    generate_matrix(config)
