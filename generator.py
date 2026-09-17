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
    "Practical household budget coach, plain English about bills, prices, and what to do next.",
    "Skeptical market observer, question hype, point out risks and who benefits.",
    "Consumer deals reporter, focus on rates, fees, refinancing, and money-saving angles for US readers.",
    "Inflation and paycheck reality voice, explain how fed/CPI/wages hit groceries, rent, and credit cards.",
]

EVERGREEN_FINANCE_SEEDS = [
    "10 year treasury yield",
    "mortgage rates today",
    "refinance mortgage rates",
    "social security cola",
    "irs tax brackets",
    "401k contribution limit",
    "credit card apr",
    "fed meeting schedule",
    "federal funds rate",
    "cpi inflation report",
    "core pce inflation",
    "average US rent",
    "30 year mortgage rate",
    "auto loan rates",
    "student loan repayment",
    "savings account apy",
    "cd rates today",
    "gas prices US average",
    "grocery inflation",
    "walmart prices",
    "costco deals",
    "credit score tips",
    "debt snowball vs avalanche",
    "fha loan requirements",
    "heloc rates",
    "unemployment rate US",
    "minimum wage by state",
    "overdraft fees banks",
    "stimulus check eligibility",
    "medicare part b premium",
    "social security retirement age",
    "roth ira income limits",
    "hsa contribution limit",
    "property tax assessment",
    "home insurance rates",
    "car insurance quotes",
    "utility bills rising",
    "egg prices US",
    "beef prices inflation",
    "used car prices",
    "airline baggage fees",
    "bank of america savings rate",
    "chase sapphire annual fee",
    "paypal credit apr",
    "layaway vs credit",
    "buy now pay later risks",
    "paycheck to paycheck budget",
    "emergency fund how much",
    "debt consolidation loan",
    "personal loan rates",
    "treasury bill auction",
    "tips inflation bonds",
    "municipal bond yields",
    "dow jones today",
    "s&p 500 outlook",
    "nasdaq composite",
    "oil price WTI",
    "gold price today",
    "dollar index dxy",
    "housing inventory US",
    "new home sales",
    "existing home sales",
    "foreclosure rates",
    "eviction moratorium status",
    "child tax credit update",
    "earned income tax credit",
    "standard deduction amount",
    "estimated tax payments",
    "capital gains tax rate",
    "crypto tax reporting irs",
    "gig worker taxes",
    "tips as taxable income",
    "401k early withdrawal penalty",
    "hardship withdrawal rules",
    "pension vs 401k",
    "annuity fees explained",
    "long term care insurance cost",
    "cobra health insurance cost",
    "short term health insurance",
    "high deductible health plan",
    "fsa vs hsa",
    "flexible spending deadline",
    "open enrollment checklist",
    "medicare advantage vs supplement",
    "social security earnings test",
    "rmd required minimum distribution",
    "qualified charitable distribution",
    "backdoor roth ira",
    "mega backdoor roth",
    "net worth calculator tips",
    "debt to income ratio mortgage",
    "points vs no points mortgage",
    "closing costs explained",
    "escrow shortage reasons",
    "pmi removal requirements",
    "va loan benefits",
    "usda rural housing loan",
    "first time homebuyer programs",
    "down payment assistance",
    "rent vs buy calculator",
    "landlord rent increase limits",
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


def fit_title(text, keyword="", min_len=50, max_len=70):
    """Keep a real title; only pad when very short. Never force keyword update suffix."""
    t = " ".join((text or "").strip().split())
    t = re.sub(r'^["\']|["\']$', "", t)
    t = re.sub(
        r"^(FOR IMMEDIATE RELEASE|BREAKING NEWS|BREAKING|HEADLINE|TITLE|UPDATE)[:\s]*",
        "",
        t,
        flags=re.IGNORECASE
    ).strip()
    # Already title-like (>=40) → do not append keyword suffix
    if len(t) < 40 and keyword:
        t = f"{t} — {keyword}".strip() if t else keyword
        t = " ".join(t.split())
    if len(t) > max_len:
        cut = t[:max_len]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        t = cut.rstrip(" ,.;:|-—")
    return t



def fit_meta_description(text, title="", keyword="", min_len=150, max_len=160):
    """Build a usable meta from title/keyword; avoid canned multi-filler spam."""
    text = (text or "").strip()
    text = re.sub(r'^["\']|["\']$', "", text).strip()
    text = " ".join(text.split())
    if not text:
        text = title or (f"Latest updates on {keyword}" if keyword else "Latest updates")
    if len(text) < min_len:
        extra = (
            f" What US readers need to know about {keyword} right now."
            if keyword else
            " What readers need to know right now."
        )
        if extra.strip().lower() not in text.lower():
            text = (text.rstrip(". ") + "." + extra).strip()
        text = " ".join(text.split())
    guard = 0
    while len(text) < min_len and guard < 4:
        text = (text + " More details inside.").strip()
        text = " ".join(text.split())
        guard += 1
    if len(text) > max_len:
        cut = text[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.rstrip(".,;:") + "…"
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


def to_html_paragraphs(text: str) -> str:
    """Turn plain article text into <p> blocks + Final Thoughts."""
    text = (text or "").strip()
    if not text:
        return "<p></p>"
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(chunks) <= 1:
        # Single newlines only — split into short paragraphs by sentences
        chunks = [
            p.strip()
            for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            if len(p.strip()) > 40
        ]
    if not chunks:
        chunks = [text]
    # If still one giant blob, hard-wrap every ~3 sentences
    if len(chunks) == 1 and len(chunks[0]) > 600:
        sents = re.split(r"(?<=[.!?])\s+", chunks[0])
        chunks = []
        buf = []
        for s in sents:
            buf.append(s)
            if len(buf) >= 3:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
    closer = chunks[-1]
    main = chunks[:-1] if len(chunks) > 1 else chunks[:]
    if len(chunks) == 1:
        main = chunks
        closer = ""
    html = "".join(f"<p>{p}</p>" for p in main if p)
    if closer and len(chunks) > 1:
        html += f"<h3>Final Thoughts</h3><p>{closer}</p>"
    return html



def fetch_single_article(persona_tuple, seed, last_updated, site_config):
    round_idx, current_persona = persona_tuple
    query = seed["query"]
    system_prompt = (
        f"You are a: {current_persona}. Write a unique, engaging viral news article for an American audience.\n"
        f"CRITICAL RULES:\n"
        f"- The VERY FIRST LINE must be the title only (no labels like TITLE: or BREAKING:).\n"
        f"- Title 50-70 characters. Never append the raw keyword. Never end with an em dash plus keyword.\n"
        f"- Curiosity-driven, specific, natural. No ALL CAPS. No repeating the same formula.\n"
        f"- Write 450-700 words. Do not write 1000+ words.\n"
        f"- Short paragraphs: 2-4 sentences each, blank line between paragraphs.\n"
        f"- Do not repeat the title in the body.\n"
        f"- After the body, add a short closing opinion (2-3 sentences).\n"
        f"- Use American English."
    )
    try:
        max_tokens = int(site_config.get("max_tokens", 900))
        min_body_chars = int(site_config.get("min_body_chars", 400))
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a viral article about: {query}"}
            ],
            max_tokens=max_tokens,
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
        clean_title = fit_title(raw_title, keyword=query, min_len=50, max_len=70)

        body_lines = lines[1:]
        # Model often repeats the title on the second line
        if body_lines:
            maybe = fit_title(body_lines[0], keyword=query, min_len=0, max_len=70)
            if maybe[:40].lower() == clean_title[:40].lower():
                body_lines = body_lines[1:]

        body_only = "\n".join(body_lines).strip()
        if len(re.sub(r"\s+", " ", body_only)) < 200:
            raw_lines = [ln for ln in content.splitlines() if ln.strip()]
            body_only = "\n".join(raw_lines[1:]).strip() or content

        # One API call: meta from title (not empty filler string)
        meta_description = fit_meta_description(
            clean_title,
            title=clean_title,
            keyword=query,
            min_len=150,
            max_len=160,
        )

        if "<p>" in body_only.lower() or "final thoughts" in body_only.lower():
            final_content = body_only
        else:
            final_content = to_html_paragraphs(body_only)

        image_path = get_pexels_image(query)
        if image_path:
            final_content = (
                f'<img src="{image_path}" alt="{clean_title}" '
                f'style="max-width:100%; height:auto; border-radius:8px;"><br><br>'
                + final_content
            )

        body_text = re.sub(r"<[^>]+>", " ", final_content or "")
        body_text = re.sub(r"\s+", " ", body_text).strip()
        clean_title = fit_title(clean_title, keyword=query, min_len=50, max_len=70)
        if len(clean_title) < 40 or len(body_text) < min_body_chars:
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
        # Exclude always; prefer filter hits; pad with evergreen if short of trends_limit
        kw_filter = [k.lower() for k in (config.get("keyword_filter") or []) if k]
        kw_exclude = [k.lower() for k in (config.get("keyword_exclude") or []) if k]
        evergreen = config.get("evergreen_seeds") or (
            EVERGREEN_FINANCE_SEEDS if (config.get("persona_style") or "").lower() == "financial"
            else []
        )

        def _excluded(q: str) -> bool:
            q = (q or "").lower()
            return bool(kw_exclude and any(x in q for x in kw_exclude))

        def _filter_hit(q: str) -> bool:
            q = (q or "").lower()
            if not kw_filter:
                return True
            return any(x in q for x in kw_filter)

        before = len(trending_seeds)
        trending_seeds = [
            s for s in trending_seeds
            if (s.get("query") or "").strip() and not _excluded(s.get("query", ""))
        ]
        hits = [s for s in trending_seeds if _filter_hit(s.get("query", ""))]
        rest = [s for s in trending_seeds if not _filter_hit(s.get("query", ""))]
        # Prefer filter hits, then other non-excluded live trends
        ordered = hits + rest
        print(
            f"🧹 Filter: {before} → live {len(ordered)} "
            f"(hits={len(hits)} other={len(rest)} exclude={len(kw_exclude)})"
        )

        seeds = ordered[:trends_limit]
        if len(seeds) < trends_limit and evergreen:
            have = {(s.get("query") or "").lower() for s in seeds}
            for q in evergreen:
                if len(seeds) >= trends_limit:
                    break
                ql = q.lower()
                if ql in have or _excluded(q):
                    continue
                seeds.append({
                    "query": q,
                    "search_volume": 0,
                    "increase": 0,
                    "source": "evergreen",
                })
                have.add(ql)
            print(f"🌿 Evergreen pad → {len(seeds)}/{trends_limit} seeds")

        if len(seeds) < 5:
            print("❌ Too few seeds after filter/exclude/evergreen — check config + Trends JSON")
            sys.exit(1)

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
