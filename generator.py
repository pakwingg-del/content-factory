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
    except:
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
    except:
        pass
    return None


def fit_meta_description(text, title="", keyword="", min_len=150, max_len=160):
    """強制 meta description 落喺 150–160 字元（Bing 建議）。"""
    text = (text or "").strip()
    text = re.sub(r'^["\']|["\']$', "", text).strip()
    text = " ".join(text.split())

    # 太短：用 title / keyword 補自然英文句子
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
        # 避免重複硬塞同一句
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

    # 最後再保證唔短過 min（極端情況）
    if len(text) < min_len:
        pad = " More details inside."
        text = (text + pad)[:max_len]

    return text


def fetch_single_article(persona_tuple, seed, last_updated, site_config):
    round_idx, current_persona = persona_tuple
    query = seed["query"]
    system_prompt = (
        f"You are a: {current_persona}. Write a unique, engaging viral news article for an American audience.\n"
        f"CRITICAL RULES:\n"
        f"- The VERY FIRST LINE must be the title only (no labels like TITLE: or BREAKING:).\n"
        f"- Title must be 55-90 characters, curiosity-driven, specific, and natural (avoid repeating the same formula).\n"
        f"- Write the main article body (800-1100 words).\n"
        f"- Do NOT write a conclusion yet.\n"
        f"- Use American English."
    )
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a viral article about: {query}"}
            ],
            max_tokens=1300,
            temperature=0.85
        )
        content = completion.choices[0].message.content.strip()
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        raw_title = lines[0] if lines else query
        clean_title = re.sub(
            r"^(FOR IMMEDIATE RELEASE|BREAKING NEWS|BREAKING|HEADLINE|TITLE|UPDATE)[:\s]*",
            "", raw_title, flags=re.IGNORECASE
        ).strip()
        clean_title = re.sub(r'^["\']|["\']$', "", clean_title).strip()
        if len(clean_title) < 35:
            clean_title = f"What's Really Happening with {query} Right Now in America"

        # Meta Description — 明確要求 150–160
        meta_prompt = (
            f"Write ONE unique SEO meta description for a news article.\n"
            f"Title: \"{clean_title}\"\n"
            f"Topic: {query}\n"
            f"STRICT RULES:\n"
            f"- Length MUST be between 150 and 160 characters including spaces\n"
            f"- Count carefully before answering\n"
            f"- Include the topic naturally\n"
            f"- Click-worthy but not spammy\n"
            f"- American English\n"
            f"- Output ONLY the description text, no quotes, no labels"
        )
        meta_completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": meta_prompt}],
            max_tokens=100,
            temperature=0.7
        )
        meta_description = fit_meta_description(
            meta_completion.choices[0].message.content.strip(),
            title=clean_title,
            keyword=query,
            min_len=150,
            max_len=160
        )

        # Human Touch Opinion
        opinion_prompt = (
            f"Based on the article about '{query}', write 2-3 insightful sentences as a personal opinion and conclusion. "
            f"Sound like a real experienced journalist."
        )
        opinion_completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": opinion_prompt}],
            max_tokens=180,
            temperature=0.9
        )
        opinion = opinion_completion.choices[0].message.content.strip()
        final_content = content + "\n\n<h3>Final Thoughts</h3>\n<p>" + opinion + "</p>"

        image_path = get_pexels_image(query)
        if image_path:
            final_content = (
                f'<img src="{image_path}" alt="{clean_title}" '
                f'style="max-width:100%; height:auto; border-radius:8px;"><br><br>' + final_content
            )

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
    except:
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

    try:
        response = requests.get(trends_url, timeout=20)
        response.raise_for_status()
        data = response.json()
        trending_seeds = data.get("trending_seeds", [])
        trending_seeds.sort(
            key=lambda x: (x.get("increase", 0), x.get("search_volume", 0)),
            reverse=True
        )
        seeds = trending_seeds[:trends_limit]
        print(f"✅ Loaded Top {len(seeds)} trends")
    except Exception as e:
        print(f"❌ Error fetching trends: {e}")
        sys.exit(1)

    all_articles = []
    MAX_WORKERS = int(config.get("max_workers", 40))
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

    print(f"✅ Generated {len(all_articles)} articles")

    # 抽樣顯示 meta 長度（方便你 check Bing 要求）
    for sample in all_articles[:3]:
        md = sample.get("meta_description", "")
        print(f"📝 Meta sample ({len(md)} chars): {md}")

    # ==================== D1 Injection ====================
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

        meta = fit_meta_description(
            article.get("meta_description", ""),
            title=article.get("title", ""),
            keyword=article.get("keyword", "")
        )

        sql = """INSERT OR REPLACE INTO articles
                 (title, keyword, body, persona_id, persona_type, search_volume, created_at, url_slug, meta_description)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);"""
        params = [
            article["title"],
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
