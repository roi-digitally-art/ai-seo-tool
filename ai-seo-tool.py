# app.py
"""
AI SEO Site Auditor (Streamlit)
- Crawl same-domain pages (async)
- Compute AI-SEO readiness per page
- Show simple 5-column report
- Gate PDF download behind a lead form
- Uses Manrope font and shows checklist explanations
"""

from __future__ import annotations
import asyncio
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple, Set, Any
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser

import aiohttp
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# -------------------- CONFIG --------------------
# Safety caps and concurrency defaults - you can change in the sidebar UI
DEFAULT_CONCURRENCY = 30      # number of parallel requests
DEFAULT_MAX_PAGES = 1000      # safety cap to avoid runaway crawling
REQUEST_TIMEOUT = 12
POLITE_DELAY_SECONDS = 0.02   # tiny delay between requests per worker (async)
USER_AGENT = "AISEOAuditor/1.0 (+https://yourcompany.example; contact=hello@yourcompany.example)"

# Scoring weights (sum = 100)
WEIGHTS = {
    "structured_data": 18,
    "conversational_content": 18,
    "faq_presence": 12,
    "content_depth": 14,
    "internal_links": 8,
    "title_meta": 10,
    "entities_context": 10,
    "authority_signals": 10,
}  # sum 100

# Minimum words for content depth
MIN_WORDS_MAIN = 300

# -------------------- HELPERS --------------------

@dataclass
class PageResult:
    url: str
    status: int
    title: str
    meta_description: str
    word_count: int
    jsonld_types: List[str]
    faq_present: bool
    howto_present: bool
    internal_links: int
    external_links: int
    images: int
    images_missing_alt: int
    issues: List[str]
    score_components: Dict[str, float]  # per-criterion scores
    final_score: int


def absolutize(base: str, href: str) -> str:
    if not href:
        return ""
    href = href.strip()
    href = urljoin(base, href)
    href, _ = urldefrag(href)
    return href


def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.scheme in {"http", "https"} and pa.netloc == pb.netloc


def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_jsonld_types(soup: BeautifulSoup) -> List[str]:
    types = []
    for tag in soup.find_all("script", type=lambda v: v and "ld+json" in v):
        try:
            import json
            data = json.loads(tag.string or "{}")
            def collect(d):
                if isinstance(d, dict):
                    t = d.get("@type")
                    if isinstance(t, str):
                        types.append(t)
                    elif isinstance(t, list):
                        for x in t:
                            if isinstance(x, str):
                                types.append(x)
                    for v in d.values():
                        collect(v)
                elif isinstance(d, list):
                    for x in d:
                        collect(x)
            collect(data)
        except Exception:
            continue
    # normalize
    return sorted(list(set([t for t in types if isinstance(t, str)])))


def extract_questions_and_headings(soup: BeautifulSoup) -> List[str]:
    q = []
    text = clean_text(soup)
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.endswith("?") or re.match(r"^(who|what|why|how|when|where)\b", s, re.I):
            if 6 <= len(s) <= 220:
                q.append(s)
    for tag in soup.find_all(["h1", "h2", "h3"]):
        t = tag.get_text(strip=True)
        if 6 <= len(t) <= 160:
            q.append(t)
    # unique
    seen = set()
    out = []
    for s in q:
        k = re.sub(r"\s+", " ", s.lower())
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:300]


# Simple entity heuristic: presence of capitalized multi-word tokens or named entities
def detect_entities(text: str) -> int:
    # count likely entities: sequences of 2+ capitalized words (e.g. "ROI Digitally")
    ents = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b", text)
    # also count presence of Brand-like tokens
    return len(set(ents))


# -------------------- PAGE ANALYSIS --------------------

def analyze_html(url: str, html: str, root_netloc: str) -> PageResult:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Title & meta
    t = soup.find("title")
    title = (t.get_text(strip=True) if t else "")
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if md and md.get("content"):
        meta_description = md.get("content", "").strip()
    else:
        ogd = soup.find("meta", attrs={"property": re.compile("og:description", re.I)})
        meta_description = ogd.get("content", "").strip() if ogd else ""

    text = clean_text(soup)
    word_count = len(re.findall(r"\w+", text))

    # Links
    internal_links = external_links = 0
    parsed_root = root_netloc
    for a in soup.find_all("a", href=True):
        href = absolutize(url, a.get("href"))
        if not href:
            continue
        try:
            hn = urlparse(href).netloc
            if hn == parsed_root:
                internal_links += 1
            else:
                external_links += 1
        except Exception:
            continue

    # Images
    imgs = soup.find_all("img")
    images = len(imgs)
    images_missing_alt = sum(1 for i in imgs if not (i.get("alt") or "").strip())

    # JSON-LD
    jsonld_types = parse_jsonld_types(soup)
    faq_present = "FAQPage" in jsonld_types or any("faq" in (t.lower() if isinstance(t, str) else "") for t in jsonld_types)
    howto_present = "HowTo" in jsonld_types

    # prompts/questions
    prompts = extract_questions_and_headings(soup)
    prompts_found = len(prompts)

    # Entities
    entity_count = detect_entities(text)

    # Issues list
    issues = []
    if not title:
        issues.append("Missing title")
    if title and not (30 <= len(title) <= 65):
        issues.append(f"Title length {len(title)} chars (ideal 30–65)")
    if not meta_description:
        issues.append("Missing meta description")
    if meta_description and not (70 <= len(meta_description) <= 160):
        issues.append(f"Meta description length {len(meta_description)} chars (ideal 70–160)")
    if word_count < MIN_WORDS_MAIN:
        issues.append(f"Thin content ({word_count} words)")
    if images > 0 and images_missing_alt > 0:
        issues.append(f"{images_missing_alt}/{images} images missing alt text")
    if internal_links < 2:
        issues.append(f"Low internal linking ({internal_links})")
    if not jsonld_types:
        issues.append("No JSON-LD structured data detected")

    # Scoring components (each 0..1 then weighted)
    comp = {}

    # 1. structured_data: presence of helpful jsonld types like FAQPage, HowTo, Product, WebSite
    sd_score = 1.0 if any(t in {"FAQPage", "HowTo", "Product", "WebSite", "Article", "BlogPosting"} for t in jsonld_types) else (0.6 if jsonld_types else 0.0)
    comp["structured_data"] = sd_score * 100

    # 2. conversational_content: heuristic - presence of FAQs, headings that are short answers, and natural sentences
    conv_score = 0.0
    if prompts_found >= 3 and word_count >= MIN_WORDS_MAIN:
        conv_score = 1.0
    elif prompts_found >= 1 and word_count >= 200:
        conv_score = 0.6
    comp["conversational_content"] = conv_score * 100

    # 3. faq_presence
    comp["faq_presence"] = (1.0 if faq_present else 0.0) * 100

    # 4. content_depth
    cd = 1.0 if word_count >= MIN_WORDS_MAIN else max(0.0, word_count / MIN_WORDS_MAIN)
    comp["content_depth"] = cd * 100

    # 5. internal linking
    il = 1.0 if internal_links >= 4 else (0.6 if internal_links >= 2 else 0.0)
    comp["internal_links"] = il * 100

    # 6. title_meta quality
    tm = 1.0 if (30 <= len(title) <= 65 and 70 <= len(meta_description) <= 160) else 0.0
    comp["title_meta"] = tm * 100

    # 7. entities & context
    ent = 1.0 if entity_count >= 2 else (0.5 if entity_count >= 1 else 0.0)
    comp["entities_context"] = ent * 100

    # 8. authority signals (heuristic: presence of author, about, contact pages, external references)
    authority = 0.0
    # quick checks: contact/about links existence in internal links count and page text
    if re.search(r"\b(contact|about|team|author|privacy|terms)\b", text, re.I):
        authority = 1.0
    comp["authority_signals"] = authority * 100

    # Combine weighted score
    total_weight = sum(WEIGHTS.values())
    weighted_sum = 0.0
    for key, w in WEIGHTS.items():
        val = comp.get(key, 0.0)
        weighted_sum += (val / 100.0) * w
    final_score = int(round((weighted_sum / total_weight) * 100))

    return PageResult(
        url=url,
        status=200,
        title=title,
        meta_description=meta_description,
        word_count=word_count,
        jsonld_types=jsonld_types,
        faq_present=faq_present,
        howto_present=howto_present,
        internal_links=internal_links,
        external_links=external_links,
        images=images,
        images_missing_alt=images_missing_alt,
        issues=issues,
        score_components=comp,
        final_score=final_score,
    )


# -------------------- ASYNC CRAWLER --------------------

class AsyncCrawler:
    def __init__(self, start_url: str, max_pages: int = DEFAULT_MAX_PAGES, concurrency: int = DEFAULT_CONCURRENCY):
        self.start_url = start_url
        parsed = urlparse(start_url)
        self.base = f"{parsed.scheme}://{parsed.netloc}"
        self.root_netloc = parsed.netloc
        self.seen: Set[str] = set()
        self.to_visit = asyncio.Queue()
        self.to_visit.put_nowait(start_url)
        self.max_pages = max_pages
        self.headers = {"User-Agent": USER_AGENT}
        self.robots = robotparser.RobotFileParser()
        self.robots.set_url(urljoin(self.base, "/robots.txt"))
        try:
            self.robots.read()
        except Exception:
            pass
        self.concurrency = concurrency
        self.results_html: Dict[str, str] = {}
        self.session = None

    async def fetch(self, url: str) -> Tuple[str, str]:
        # returns (url, text) or (url, "")
        try:
            async with self.session.get(url, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status != 200:
                    return url, ""
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                    return url, ""
                text = await resp.text()
                return url, text
        except Exception:
            return url, ""

    async def worker(self, sem: asyncio.Semaphore):
        while True:
            try:
                url = await asyncio.wait_for(self.to_visit.get(), timeout=1.0)
            except asyncio.TimeoutError:
                return
            if url in self.seen:
                self.to_visit.task_done()
                continue
            if len(self.seen) >= self.max_pages:
                self.to_visit.task_done()
                return
            # respect robots
            try:
                can_fetch = self.robots.can_fetch(USER_AGENT, url)
            except Exception:
                can_fetch = True
            if not can_fetch:
                self.seen.add(url)
                self.to_visit.task_done()
                continue
            self.seen.add(url)
            async with sem:
                await asyncio.sleep(POLITE_DELAY_SECONDS)
                fetched_url, html = await self.fetch(url)
            if html:
                # store html
                self.results_html[fetched_url] = html
                # find links to enqueue
                try:
                    soup = BeautifulSoup(html, "lxml")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = absolutize(fetched_url, a.get("href"))
                    if href and same_domain(self.base, href):
                        if href not in self.seen:
                            await self.to_visit.put(href)
            self.to_visit.task_done()

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        async with aiohttp.ClientSession(headers=self.headers, timeout=timeout, connector=connector) as sess:
            self.session = sess
            sem = asyncio.Semaphore(self.concurrency)
            workers = [asyncio.create_task(self.worker(sem)) for _ in range(self.concurrency)]
            await self.to_visit.join()
            for w in workers:
                w.cancel()
            # allow cancellations to finish
            await asyncio.gather(*workers, return_exceptions=True)

    def crawl(self):
        asyncio.run(self.run())
        return list(self.results_html.items())  # list of (url, html)


# -------------------- PDF REPORT --------------------

def build_pdf(site: str, page_results: List[PageResult]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    story = []

    story.append(Paragraph(f"AI SEO Audit — {site}", h1))
    story.append(Spacer(1, 6))

    total_pages = len(page_results)
    avg_score = int(sum(p.final_score for p in page_results) / total_pages) if total_pages else 0
    story.append(Paragraph(f"Pages crawled: <b>{total_pages}</b> · Overall AI SEO Score: <b>{avg_score}/100</b>", normal))
    story.append(Spacer(1, 8))

    # Top issues summary
    top = sorted(page_results, key=lambda r: len(r.issues), reverse=True)[:12]
    data = [["URL", "Score", "Issues (top)"]]
    for p in top:
        data.append([p.url, str(p.final_score), "; ".join(p.issues[:3]) or "None"])
    tbl = Table(data, colWidths=[90*mm, 20*mm, 70*mm])
    tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    # Detailed page entries (top 10 by score or by issues)
    story.append(Paragraph("Page Summaries", h2))
    story.append(Spacer(1, 6))
    for p in page_results[:20]:
        story.append(Paragraph(f"<b>{p.url}</b> — Score: {p.final_score}/100", normal))
        comps = ", ".join(f"{k}: {int(v)}" for k, v in p.score_components.items())
        story.append(Paragraph(f"Components: {comps}", normal))
        if p.issues:
            story.append(Paragraph(f"Issues: {'; '.join(p.issues)}", normal))
        else:
            story.append(Paragraph("No critical AI-SEO issues detected.", normal))
        story.append(Spacer(1, 6))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="AI SEO Auditor", page_icon="🤖", layout="wide")
# Inject Manrope font and some CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    .stApp { padding-top: 12px; }
    .small-muted { color: #6b7280; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI SEO Site Auditor (Simple & Focused)")
st.markdown("""
**Disclaimer:** This tool is a heuristic AI-SEO readiness checker and can be improved with feedback.  
If you think results are not accurate or want a deeper review, please book a meeting with us below.
<!-- Calendly inline widget begin -->
<div class="calendly-inline-widget" style="min-width: 320px; height: 450px;" data-url="https://calendly.com/tracy-roidigitally/30min"></div>
<script type="text/javascript" src="https://assets.calendly.com/assets/external/widget.js" async></script>
<!-- Calendly inline widget end -->
""", unsafe_allow_html=True)

st.markdown("### Quick Checklist (what we check and why) — short plain explanations")
st.markdown("""
- **Structured Data (Schema markup on key pages):** Labels like FAQPage/HowTo/Product help AI extract exact answers and show richer results.
- **Clear, Conversational Content:** Content should read like an answer — short clear explanations and Q&A sections, not keyword lists.
- **FAQs / Q&A:** FAQs are easy for AI to copy as concise answers.
- **Content Depth:** Main pages should fully answer a topic (recommended ≥ 300 words).
- **Internal Linking for Context:** Link related pages so AI understands how your site connects topics.
- **Page Titles & Meta for AI Overviews:** Titles/meta should sound like answers to user questions where appropriate.
- **Entities & Context:** Use clear names (brands, places, product names) so AI knows what the content specifically refers to.
- **Fast & Mobile-first:** Technical signals indirectly help AI prefer your content.
- **Authoritative Signals (E-E-A-T):** Author bios, references, contact info improve trust.
- **No Duplicate / Thin Content:** Original and substantive content increases AI visibility.
""")

with st.sidebar:
    st.header("Run settings (advanced, optional)")
    concurrency = st.number_input("Concurrency (parallel requests)", min_value=3, max_value=200, value=40, step=1)
    max_pages = st.number_input("Safety cap – max pages to fetch (to avoid runaway crawls)", min_value=50, max_value=5000, value=1000, step=50)
    run_button = st.button("🚀 Run full-site AI SEO audit")

# Results area placeholders
results_placeholder = st.empty()
table_placeholder = st.empty()
form_placeholder = st.empty()

if run_button:
    start_url = st.text_input("Start URL (enter again to confirm)", value="")
    if not start_url:
        st.info("Enter the site URL at the top of the page to start crawling.")
        st.stop()

    # normalize
    start_url = start_url.strip()
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    # run async crawler
    with st.spinner("Crawling site (async). This may take a few seconds depending on site size..."):
        crawler = AsyncCrawler(start_url, max_pages=int(max_pages), concurrency=int(concurrency))
        items = crawler.crawl()  # returns list of (url, html)
    if not items:
        st.error("No HTML pages found or crawl blocked by robots.txt.")
        st.stop()

    # analyze pages (synchronous parsing)
    page_results: List[PageResult] = []
    root_netloc = urlparse(start_url).netloc
    pbar = st.progress(0.0)
    for i, (url, html) in enumerate(items):
        pr = analyze_html(url, html, root_netloc)
        page_results.append(pr)
        pbar.progress((i + 1) / len(items))

    # build DataFrame for table (only 5 columns)
    df_rows = []
    for p in page_results:
        structured = "Yes" if p.jsonld_types else "No"
        ai_friendly = "Yes" if (p.final_score >= 65) else "Partial" if p.final_score >= 45 else "No"
        issues_text = "; ".join(p.issues) if p.issues else "None"
        df_rows.append({
            "url": p.url,
            "score": p.final_score,
            "structured_data": structured,
            "ai_friendly_content": ai_friendly,
            "issues": issues_text
        })
    df = pd.DataFrame(df_rows)
    # Overall site score
    overall = int(round(df["score"].mean())) if len(df) else 0
    label = "Bad" if overall < 50 else ("Average" if overall < 80 else "Good")

    results_placeholder.markdown(f"## Results — Overall AI SEO Score: **{overall}/100** — **{label}**")
    results_placeholder.markdown("Use the table below to review page readiness. Only the most relevant AI-SEO checks are shown.")
    # show the five-column table
    table_placeholder.dataframe(df.sort_values("score", ascending=False).reset_index(drop=True), use_container_width=True)

    # require lead capture form to enable PDF
    with form_placeholder.form("lead_form"):
        st.subheader("Get your downloadable PDF report")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Full name")
        email = c2.text_input("Business email")
        company = c3.text_input("Company")
        agree = st.checkbox("I agree to be contacted about this audit", value=False)
        submitted = st.form_submit_button("Generate & Download PDF")

    if submitted:
        if not name or not email or not company or not agree:
            st.error("Please fill all fields and accept contact permission to download the report.")
        else:
            with st.spinner("Building PDF report..."):
                pdf_bytes = build_pdf(urlparse(start_url).netloc, page_results)
            st.success("PDF generated — click below to download.")
            st.download_button("⬇️ Download AI SEO PDF report", data=pdf_bytes, file_name="ai_seo_audit.pdf", mime="application/pdf")

