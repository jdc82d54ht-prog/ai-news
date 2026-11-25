# app.py
# ==========================================
# [설정 영역] API 키와 뉴스 카테고리를 여기서 수정하세요
# ==========================================
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # 키를 넣지 않아도 RSS 기사 목록은 정상 표시됩니다.

# 카테고리별 최대 수집 기사 수 (5~10개 권장)
MAX_ARTICLES = 6

# 분야별 RSS 피드 설정 (원하는 분야를 자유롭게 추가/삭제하세요)
CATEGORIES = [
    {
        "id": "ai_it",
        "name": "AI/IT",
        "rss_url": "https://news.google.com/rss/search?q=AI+OR+인공지능+OR+IT기술&hl=ko&gl=KR&ceid=KR:ko"
    },
    {
        "id": "science",
        "name": "과학",
        "rss_url": "https://news.google.com/rss/search?q=과학+OR+우주+OR+바이오&hl=ko&gl=KR&ceid=KR:ko"
    },
    {
        "id": "economy",
        "name": "경제",
        "rss_url": "https://news.google.com/rss/search?q=경제+OR+증시+OR+금리&hl=ko&gl=KR&ceid=KR:ko"
    },
    {
        "id": "global",
        "name": "글로벌 트렌드",
        "rss_url": "https://news.google.com/rss/search?q=글로벌+트렌드+OR+국제+뉴스&hl=ko&gl=KR&ceid=KR:ko"
    }
]

# ==========================================
# 백엔드 로직 (Flask + RSS + Gemini API)
# ==========================================
import re
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
import feedparser
from google import genai
from google.genai import types
from pydantic import BaseModel

app = Flask(__name__)

# 유효한 API 키가 입력된 경우에만 Gemini 클라이언트 초기화
has_valid_key = bool(
    GEMINI_API_KEY
    and GEMINI_API_KEY.strip()
    and GEMINI_API_KEY.strip() != "여기에_키_입력"
)
client = genai.Client(api_key=GEMINI_API_KEY.strip()) if has_valid_key else None

# 한국 표준시 (KST)
KST = timezone(timedelta(hours=9))

# 인메모리 캐시 { category_id: { "updated_at": "...", "articles": [...] } }
NEWS_CACHE = {}

# 뉴스 요약 AI 시스템 프롬프트
SYSTEM_PROMPT = """너는 매일 아침 핵심 뉴스를 정리해 주는 전문 에디터야. 아래 제공되는 뉴스 기사 원문(또는 제목과 본문)을 읽고, 독자가 10초 만에 내용을 파악할 수 있도록 정리해 줘.

[규칙]
1. 반드시 한국어로 작성하고, 객관적이고 간결한 해요체 또는 명사형으로 끝맺어.
2. 불필요한 수식어나 낚시성 표현은 모두 제거하고 팩트와 핵심 변화 위주로만 요약해.
3. 출력은 웹페이지에서 바로 렌더링할 수 있도록 반드시 아래 JSON 형식만 반환해. (마크다운 코드 블록 외 다른 설명 금지)

{
  "category": "지정된 분야명",
  "title": "핵심을 직관적으로 드러내는 재가공 제목 (30자 이내)",
  "summary": [
    "핵심 내용 1줄 (배경 또는 주요 사건)",
    "핵심 내용 2줄 (구체적 수치나 기술적 포인트)",
    "핵심 내용 3줄 (향후 전망 또는 시사점)"
  ],
  "keywords": ["키워드1", "키워드2", "키워드3"]
}"""


class ArticleSummary(BaseModel):
    category: str
    title: str
    summary: list[str]
    keywords: list[str]


def clean_html(raw_html: str) -> str:
    """RSS 설명 텍스트에서 HTML 태그 제거"""
    return re.sub(r"<[^>]+>", "", raw_html or "").strip()


def fetch_rss_articles(rss_url: str, limit: int = MAX_ARTICLES):
    """RSS 피드에서 최신 기사 수집 및 최신순 정렬"""
    feed = feedparser.parse(rss_url)
    articles = []
    today_str = datetime.now(KST).strftime("%Y-%m-%d")

    for entry in feed.entries[: limit * 2]:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            dt_utc = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            dt_kst = dt_utc.astimezone(KST)
            pub_date_str = dt_kst.strftime("%Y-%m-%d %H:%M")
            is_today = dt_kst.strftime("%Y-%m-%d") == today_str
            sort_ts = dt_kst.timestamp()
        else:
            pub_date_str = "방금 전"
            is_today = True
            sort_ts = datetime.now(KST).timestamp()

        raw_title = entry.get("title", "제목 없음")
        source = entry.get("source", {}).get("title", "")
        if not source and " - " in raw_title:
            title, source = raw_title.rsplit(" - ", 1)
        else:
            title = raw_title
            source = source or "RSS 뉴스"

        snippet = clean_html(entry.get("summary", entry.get("description", "")))

        articles.append({
            "title": title.strip(),
            "original_title": title.strip(),
            "source": source.strip(),
            "link": entry.get("link", "#"),
            "published": pub_date_str,
            "is_today": is_today,
            "sort_ts": sort_ts,
            "snippet": snippet[:500]
        })

    articles.sort(key=lambda x: x["sort_ts"], reverse=True)
    return articles[:limit]


def summarize_single_article(article: dict, category_name: str) -> dict:
    """단일 기사에 대해 시스템 프롬프트를 적용하여 Gemini API 호출"""
    if not client:
        preview = article["snippet"][:110] + "..." if len(article["snippet"]) > 110 else article["snippet"]
        article["summary_lines"] = [
            preview if preview else "RSS 요약 텍스트가 없는 기사입니다.",
            "AI 3줄 요약을 켜려면 app.py 상단에 GEMINI_API_KEY를 입력해 주세요.",
            "카드를 클릭하면 원문 기사로 바로 이동합니다."
        ]
        article["keywords"] = [category_name, "RSS원문", "API키미설정"]
        return article

    user_content = (
        f"지정된 분야명: {category_name}\n"
        f"기사 제목: {article['original_title']}\n"
        f"기사 본문(발췌): {article['snippet']}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ArticleSummary,
                temperature=0.2,
            ),
        )
        ai_data = json.loads(response.text)

        article["title"] = ai_data.get("title", article["original_title"])
        article["summary_lines"] = ai_data.get("summary", [])[:3]
        article["keywords"] = ai_data.get("keywords", [])[:3]
    except Exception as e:
        print(f"[Gemini API 오류 - {article['original_title']}] {e}")
        article["summary_lines"] = [
            "AI 요약 중 일시적인 오류가 발생했습니다.",
            f"원문 제목: {article['original_title']}",
            "잠시 후 '최신 뉴스 업데이트'를 눌러 다시 시도해 주세요."
        ]
        article["keywords"] = ["오류", "재시도", "뉴스"]

    return article


def summarize_with_gemini(articles: list[dict], category_name: str) -> list[dict]:
    """수집된 기사 목록을 병렬 처리로 빠르게 요약"""
    if not articles:
        return []

    with ThreadPoolExecutor(max_workers=len(articles)) as executor:
        results = list(
            executor.map(lambda a: summarize_single_article(a, category_name), articles)
        )
    return results


# ==========================================
# 프론트엔드 단일 HTML + CSS + JS 템플릿
# ==========================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>오늘의 AI 뉴스 브리핑</title>
  <style>
    :root {
      --bg-color: #f4f6f9;
      --surface-color: #ffffff;
      --text-primary: #111827;
      --text-secondary: #4b5563;
      --text-muted: #6b7280;
      --border-color: #e5e7eb;
      --accent-color: #2563eb;
      --accent-hover: #1d4ed8;
      --tag-bg: #eff6ff;
      --tag-text: #1d4ed8;
      --shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      --shadow-hover: 0 10px 24px rgba(0, 0, 0, 0.1);
      --skeleton-base: #e5e7eb;
      --skeleton-highlight: #f3f4f6;
    }

    [data-theme="dark"] {
      --bg-color: #0f172a;
      --surface-color: #1e293b;
      --text-primary: #f8fafc;
      --text-secondary: #cbd5e1;
      --text-muted: #94a3b8;
      --border-color: #334155;
      --accent-color: #3b82f6;
      --accent-hover: #60a5fa;
      --tag-bg: rgba(59, 130, 246, 0.15);
      --tag-text: #93c5fd;
      --shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      --shadow-hover: 0 10px 24px rgba(0, 0, 0, 0.45);
      --skeleton-base: #334155;
      --skeleton-highlight: #475569;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-primary);
      line-height: 1.6;
      transition: background-color 0.25s ease, color 0.25s ease;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }

    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-color);
    }

    .brand h1 {
      font-size: 1.75rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .date-display {
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 16px;
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border-color);
      background-color: var(--surface-color);
      color: var(--text-primary);
      transition: all 0.2s ease;
    }

    .btn:hover {
      border-color: var(--accent-color);
    }

    .btn-primary {
      background-color: var(--accent-color);
      color: #ffffff;
      border-color: var(--accent-color);
    }

    .btn-primary:hover {
      background-color: var(--accent-hover);
      border-color: var(--accent-hover);
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .tabs-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 24px;
    }

    .tabs {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }

    .tab-btn {
      padding: 9px 18px;
      border-radius: 999px;
      border: 1px solid var(--border-color);
      background-color: var(--surface-color);
      color: var(--text-secondary);
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s ease;
    }

    .tab-btn:hover {
      border-color: var(--accent-color);
      color: var(--accent-color);
    }

    .tab-btn.active {
      background-color: var(--accent-color);
      color: #ffffff;
      border-color: var(--accent-color);
    }

    .update-status {
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .news-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 20px;
    }

    .news-card {
      display: flex;
      flex-direction: column;
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 22px;
      text-decoration: none;
      color: inherit;
      box-shadow: var(--shadow);
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .news-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-hover);
      border-color: var(--accent-color);
    }

    .card-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 12px;
    }

    .source-badge {
      font-weight: 700;
      color: var(--accent-color);
    }

    .card-title {
      font-size: 1.12rem;
      font-weight: 700;
      line-height: 1.45;
      margin-bottom: 14px;
      color: var(--text-primary);
    }

    .summary-list {
      list-style: none;
      margin-bottom: 18px;
      flex-grow: 1;
    }

    .summary-list li {
      position: relative;
      padding-left: 14px;
      font-size: 0.92rem;
      color: var(--text-secondary);
      margin-bottom: 8px;
      line-height: 1.5;
    }

    .summary-list li::before {
      content: "•";
      position: absolute;
      left: 0;
      color: var(--accent-color);
      font-weight: bold;
    }

    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding-top: 12px;
      border-top: 1px solid var(--border-color);
    }

    .tag {
      font-size: 0.8rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      background-color: var(--tag-bg);
      color: var(--tag-text);
    }

    .skeleton-card {
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 22px;
      height: 280px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .skeleton-line {
      height: 16px;
      border-radius: 6px;
      background: linear-gradient(90deg, var(--skeleton-base) 25%, var(--skeleton-highlight) 50%, var(--skeleton-base) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.4s infinite;
    }

    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    .spin {
      display: inline-block;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      100% { transform: rotate(360deg); }
    }

    @media (max-width: 600px) {
      .news-grid {
        grid-template-columns: 1fr;
      }
      .brand h1 {
        font-size: 1.4rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>📰 오늘의 AI 뉴스 브리핑</h1>
        <div class="date-display" id="todayDate"></div>
      </div>
      <div class="header-actions">
        <button class="btn" id="themeToggle" aria-label="테마 전환">🌙 다크 모드</button>
        <button class="btn btn-primary" id="refreshBtn">
          <span id="refreshIcon">🔄</span> 최신 뉴스 업데이트
        </button>
      </div>
    </header>

    <div class="tabs-bar">
      <div class="tabs" id="categoryTabs"></div>
      <div class="update-status" id="updateStatus"></div>
    </div>

    <main class="news-grid" id="newsGrid"></main>
  </div>

  <script>
    let currentCategoryId = null;

    function renderTodayDate() {
      const now = new Date();
      const options = { year: "numeric", month: "long", day: "numeric", weekday: "long" };
      document.getElementById("todayDate").textContent = now.toLocaleDateString("ko-KR", options);
    }

    function initTheme() {
      const savedTheme = localStorage.getItem("theme") || "light";
      applyTheme(savedTheme);

      document.getElementById("themeToggle").addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem("theme", next);
      });
    }

    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      document.getElementById("themeToggle").textContent =
        theme === "dark" ? "☀️ 라이트 모드" : "🌙 다크 모드";
    }

    function renderSkeletons(count = 6) {
      const grid = document.getElementById("newsGrid");
      grid.innerHTML = Array.from({ length: count })
        .map(
          () => `
          <div class="skeleton-card">
            <div class="skeleton-line" style="width: 40%;"></div>
            <div class="skeleton-line" style="width: 90%; height: 22px;"></div>
            <div class="skeleton-line" style="width: 100%;"></div>
            <div class="skeleton-line" style="width: 95%;"></div>
            <div class="skeleton-line" style="width: 80%;"></div>
          </div>`
        )
        .join("");
    }

    async function loadCategories() {
      const res = await fetch("/api/categories");
      const categories = await res.json();
      const tabsEl = document.getElementById("categoryTabs");
      tabsEl.innerHTML = "";

      categories.forEach((cat, idx) => {
        const btn = document.createElement("button");
        btn.className = "tab-btn" + (idx === 0 ? " active" : "");
        btn.textContent = cat.name;
        btn.dataset.id = cat.id;
        btn.addEventListener("click", () => {
          document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          loadNews(cat.id, false);
        });
        tabsEl.appendChild(btn);
      });

      if (categories.length > 0) {
        loadNews(categories[0].id, false);
      }
    }

    async function loadNews(categoryId, forceRefresh = false) {
      currentCategoryId = categoryId;
      const refreshBtn = document.getElementById("refreshBtn");
      const refreshIcon = document.getElementById("refreshIcon");
      const statusEl = document.getElementById("updateStatus");

      refreshBtn.disabled = true;
      refreshIcon.classList.add("spin");
      statusEl.textContent = "최신 기사를 불러오고 분석 중입니다...";
      renderSkeletons();

      try {
        const res = await fetch(`/api/news/${categoryId}?force=${forceRefresh}`);
        const data = await res.json();

        if (!res.ok) throw new Error(data.error || "뉴스를 불러오지 못했습니다.");

        renderCards(data.articles);
        statusEl.textContent = `마지막 업데이트: ${data.updated_at}`;
      } catch (err) {
        document.getElementById("newsGrid").innerHTML =
          `<p style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-muted);">오류: ${err.message}</p>`;
        statusEl.textContent = "업데이트 실패";
      } finally {
        refreshBtn.disabled = false;
        refreshIcon.classList.remove("spin");
      }
    }

    function renderCards(articles) {
      const grid = document.getElementById("newsGrid");
      if (!articles || articles.length === 0) {
        grid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; padding: 40px;">표시할 최신 기사가 없습니다.</p>`;
        return;
      }

      grid.innerHTML = articles
        .map(
          (article) => `
          <a href="${article.link}" target="_blank" rel="noopener noreferrer" class="news-card">
            <div class="card-meta">
              <span class="source-badge">${escapeHtml(article.source)}</span>
              <span>${escapeHtml(article.published)}</span>
            </div>
            <h2 class="card-title">${escapeHtml(article.title)}</h2>
            <ul class="summary-list">
              ${(article.summary_lines || [])
                .filter((line) => line.trim() !== "")
                .map((line) => `<li>${escapeHtml(line)}</li>`)
                .join("")}
            </ul>
            <div class="tags">
              ${(article.keywords || [])
                .map((kw) => `<span class="tag">#${escapeHtml(kw)}</span>`)
                .join("")}
            </div>
          </a>`
        )
        .join("");
    }

    function escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text || "";
      return div.innerHTML;
    }

    document.getElementById("refreshBtn").addEventListener("click", () => {
      if (currentCategoryId) {
        loadNews(currentCategoryId, true);
      }
    });

    renderTodayDate();
    initTheme();
    loadCategories();
  </script>
</body>
</html>"""


# ==========================================
# 라우트 설정 및 서버 실행
# ==========================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/categories")
def get_categories():
    return jsonify([{"id": c["id"], "name": c["name"]} for c in CATEGORIES])


@app.route("/api/news/<category_id>")
def get_news(category_id):
    force_refresh = request.args.get("force", "false").lower() == "true"

    category = next((c for c in CATEGORIES if c["id"] == category_id), None)
    if not category:
        return jsonify({"error": "존재하지 않는 카테고리입니다."}), 404

    if not force_refresh and category_id in NEWS_CACHE:
        return jsonify(NEWS_CACHE[category_id])

    raw_articles = fetch_rss_articles(category["rss_url"])
    enriched_articles = summarize_with_gemini(raw_articles, category["name"])

    result = {
        "category": category["name"],
        "updated_at": datetime.now(KST).strftime("%H:%M:%S"),
        "articles": enriched_articles
    }
    NEWS_CACHE[category_id] = result
    return jsonify(result)


if __name__ == "__main__":
    print("==========================================")
    print(" 오늘의 AI 뉴스 브리핑 서버가 시작되었습니다!")
    print(" 브라우저 주소창에 입력: http://127.0.0.1:5000")
    print("==========================================")
    app.run(debug=True, host="127.0.0.1", port=5000)