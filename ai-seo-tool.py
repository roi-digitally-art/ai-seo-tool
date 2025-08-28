"""
AI SEO Site Auditor – Streamlit
---------------------------------
What it does
- Crawl an entire site (same domain) with robots.txt respect
- Check AI‑SEO readiness: structured data (JSON‑LD types), FAQ/HowTo, WebSite SearchAction, metadata quality,
  indexability (meta robots, canonical), sitemap presence, internal linking, image alts, content depth
- Generate "prompt coverage" by extracting likely user prompts (questions, H1/H2) and estimating how many the site answers
  (lexical overlap). Optionally, upgrade scoring with OpenAI (if key provided)
- Show results in a dashboard, plus export a polished **PDF report**
- Gate the PDF download behind a simple lead form (name, email, company)

How to run locally
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud
    - Push app.py + requirements.txt to GitHub
    - New app → pick repo → app.py → Deploy
"""
from __future__ import annotations

import re
import time
import queue
from io import BytesIO
from typing import List, Dict, Tuple, Set
from urllib.parse import urljoin, urldefrag, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib import utils
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.lib import colors

# Optional OpenAI (for upgraded scoring)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------- Config ----------------------
DEFAULT_HEADERS = {
    "User-Agent": "AISEOAuditor/1.0 (+https://example.com; contact=support@example.com)",
}
REQUEST_TIMEOUT = 15
POLITE_DELAY = 0.6

TITLE_GOOD_RANGE = (30, 65)
DESC_GOOD_RANGE = (70, 160)
MIN_WORDS = 250

JSONLD_TYPES_OF_INTEREST = {
    "FAQPage", "HowTo", "Product", "Article", "BlogPosting", "Organization",
    "LocalBusiness", "BreadcrumbList", "Recipe", "Event", "WebSite", "Service",
}

# ---------------------- Helpers ----------------------
class Robots:
    def __init__(self, root_url: str, ua: str):
        self.rp = robotparser.RobotFileParser()
        parsed = urlparse(root_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
        except Exception:
            pass
        self.ua = ua

    def allowed(self, url: str) -> bool:
        try:
            return self.rp.can_fetch(self.ua, url)
        except Exception:
            return True


def is_html(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "text/html" in ctype or "application/xhtml+xml" in ctype


def absolutize(base: str, href: str) -> str:
    if not href:
        return ""
    href = urljoin(base, href.strip())
    href, _ = urldefrag(href)
    return href


def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.scheme in {"http", "https"} and pa.netloc == pb.netloc


def clean_text(soup: BeautifulSoup) -> str:
    for t in soup(["script", "style", "noscript", "template"]):
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ---------------------- Crawl ----------------------

def crawl_site(start_url: str, max_pages: int, max_depth: int) -> List[str]:
    start_url, _ = urldefrag(start_url)
    parsed = urlparse(start_url)
    root = f"{parsed.scheme}://{parsed.netloc}"

    robots = Robots(start_url, DEFAULT_HEADERS["User-Agent"])

    seen: Set[str] = set()
    q: queue.Queue[Tuple[str, int]] = queue.Queue()
    q.put((start_url, 0))

    pages: List[str] = []

    while not q.empty() and len(pages) < max_pages:
        url, depth = q.get()
        if url in seen or depth > max_depth:
            continue
        if not robots.allowed(url):
            continue
        try:
            time.sleep(POLITE_DELAY)
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        except Exception:
            seen.add(url)
            continue

        seen.add(url)
        if resp.status_code >= 400 or not is_html(resp):
            continue

        pages.append(url)

        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = absolutize(url, a.get("href"))
            if href and same_domain(start_url, href) and href not in seen:
                q.put((href, depth + 1))

        if len(pages) >= max_pages:
            break

    return pages

# ---------------------- Analysis ----------------------

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
                        types.extend([x for x in t if isinstance(x, str)])
                    for v in d.values():
                        collect(v)
                elif isinstance(d, list):
                    for x in d:
                        collect(x)
            collect(data)
        except Exception:
            continue
    return sorted(set([t for t in types if t in JSONLD_TYPES_OF_INTEREST]))


def extract_questions_and_headings(soup: BeautifulSoup) -> List[str]:
    q = []
    # Questions in text
    text = clean_text(soup)
    for line in text.split("\n"):
        s = line.strip()
        if s.endswith("?") or re.match(r"^(who|what|why|how|when|where)\b", s, re.I):
            if 8 <= len(s) <= 160:
                q.append(s)
    # Headings
    for tag in soup.find_all(["h1", "h2", "h3"]):
        t = tag.get_text(strip=True)
        if 8 <= len(t) <= 160:
            q.append(t)
    # Deduplicate
    seen = set()
    out = []
    for s in q:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:200]  # cap per page


def lexical_prompt_coverage(page_title: str, meta_desc: str, h1s: List[str], prompts: List[str]) -> int:
    # Simple lexical overlap as a proxy for "answerability"
    base = " ".join([page_title or "", meta_desc or "", " ".join(h1s or [])]).lower()
    base_words = set(re.findall(r"[a-z0-9]+", base))
    covered = 0
    for p in prompts:
        pw = set(re.findall(r"[a-z0-9]+", p.lower()))
        if len(base_words.intersection(pw)) >= max(2, int(0.3 * max(1, len(pw)))):
            covered += 1
    return covered


def analyze_page(url: str) -> Dict:
    row = {
        "url": url,
        "status": 0,
        "content_type": "",
        "title": "",
        "title_len": 0,
        "meta_description": "",
        "meta_description_len": 0,
        "h1_count": 0,
        "h1_texts": [],
        "canonical": "",
        "meta_robots": "",
        "word_count": 0,
        "int_links": 0,
        "ext_links": 0,
        "images": 0,
        "images_missing_alt": 0,
        "jsonld_types": [],
        "faq_present": False,
        "howto_present": False,
        "website_searchaction": False,
        "sitemap_hint": False,
        "prompts_found": 0,
        "prompts_covered": 0,
        "issues": [],
    }
    try:
        time.sleep(POLITE_DELAY)
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        row["status"] = resp.status_code
        row["content_type"] = resp.headers.get("Content-Type", "")
        if resp.status_code >= 400:
            row["issues"].append(f"HTTP {resp.status_code}")
            return row
        if not is_html(resp):
            row["issues"].append("Non-HTML resource")
            return row
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Metadata
        t = soup.find("title")
        title = (t.get_text(strip=True) if t else "")
        row["title"] = title
        row["title_len"] = len(title)

        md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        if md and md.get("content"):
            meta_desc = md.get("content", "").strip()
        else:
            ogd = soup.find("meta", attrs={"property": re.compile("og:description", re.I)})
            meta_desc = ogd.get("content", "").strip() if ogd else ""
        row["meta_description"] = meta_desc
        row["meta_description_len"] = len(meta_desc)

        h1s = [h.get_text(strip=True) for h in soup.find_all("h1") if h.get_text(strip=True)]
        row["h1_texts"] = h1s
        row["h1_count"] = len(h1s)

        can = soup.find("link", rel=lambda v: v and "canonical" in v)
        canonical = can.get("href", "").strip() if can else ""
        row["canonical"] = absolutize(url, canonical) if canonical else ""

        mr = soup.find("meta", attrs={"name": re.compile("robots", re.I)})
        row["meta_robots"] = mr.get("content", "").strip() if mr else ""

        # Content
        text = clean_text(soup)
        row["word_count"] = len(re.findall(r"\w+", text))

        # Links
        parsed_root = urlparse(url)
        for a in soup.find_all("a", href=True):
            href = absolutize(url, a.get("href"))
            if not href:
                continue
            if urlparse(href).netloc == parsed_root.netloc:
                row["int_links"] += 1
            else:
                row["ext_links"] += 1

        # Images
        imgs = soup.find_all("img")
        row["images"] = len(imgs)
        row["images_missing_alt"] = sum(1 for i in imgs if not (i.get("alt") or "").strip())

        # JSON-LD
        types = parse_jsonld_types(soup)
        row["jsonld_types"] = types
        row["faq_present"] = "FAQPage" in types
        row["howto_present"] = "HowTo" in types
        row["website_searchaction"] = "WebSite" in types  # we'll add sitemap check below

        # Prompts
        prompts = extract_questions_and_headings(soup)
        row["prompts_found"] = len(prompts)
        row["prompts_covered"] = lexical_prompt_coverage(title, meta_desc, h1s, prompts)

        # Issues
        if not title:
            row["issues"].append("Missing <title>")
        elif not (TITLE_GOOD_RANGE[0] <= len(title) <= TITLE_GOOD_RANGE[1]):
            row["issues"].append(f"Title length {len(title)} (ideal {TITLE_GOOD_RANGE[0]}–{TITLE_GOOD_RANGE[1]})")
        if not meta_desc:
            row["issues"].append("Missing meta description")
        elif not (DESC_GOOD_RANGE[0] <= len(meta_desc) <= DESC_GOOD_RANGE[1]):
            row["issues"].append(f"Meta description length {len(meta_desc)} (ideal {DESC_GOOD_RANGE[0]}–{DESC_GOOD_RANGE[1]})")
        if row["h1_count"] == 0:
            row["issues"].append("No H1 found")
        elif row["h1_count"] > 1:
            row["issues"].append(f"Multiple H1s ({row['h1_count']})")
        if not row["canonical"]:
            row["issues"].append("Missing canonical link")
        if row["meta_robots"] and any(x in row["meta_robots"].lower() for x in ["noindex", "nofollow"]):
            row["issues"].append(f"Robots meta contains: {row['meta_robots']}")
        if row["images_missing_alt"] > 0:
            row["issues"].append(f"{row['images_missing_alt']}/{row['images']} images missing alt")
        if row["word_count"] < MIN_WORDS:
            row["issues"].append(f"Thin content ({row['word_count']} words)")

    except Exception as e:
        row["issues"].append(f"Error: {e}")

    return row


def check_sitemap(start_url: str) -> bool:
    parsed = urlparse(start_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap_index.xml.gz"):
        url = base + path
        try:
            time.sleep(0.1)
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=6)
            if r.status_code == 200 and ("xml" in r.headers.get("Content-Type", "").lower() or r.text.startswith("<?xml")):
                return True
        except Exception:
            continue
    return False

# ---------------------- PDF Export ----------------------

def build_pdf(site: str, df: pd.DataFrame, ai_summary: str = "") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Heading1"].alignment = TA_LEFT
    styles["Heading2"].alignment = TA_LEFT

    story = []
    story.append(Paragraph(f"AI SEO Audit – {site}", styles["Heading1"]))
    story.append(Spacer(1, 6))

    # Summary metrics
    total_pages = len(df)
    pages_with_faq = int((df["faq_present"] == True).sum()) if total_pages else 0
    pages_with_howto = int((df["howto_present"] == True).sum()) if total_pages else 0
    avg_prompt_coverage = int(df["prompts_covered"].mean()) if total_pages else 0
    story.append(Paragraph(
        f"Pages crawled: <b>{total_pages}</b> · Pages with FAQ: <b>{pages_with_faq}</b> · Pages with HowTo: <b>{pages_with_howto}</b> · Avg prompts covered/page: <b>{avg_prompt_coverage}</b>",
        styles["Normal"],
    ))
    story.append(Spacer(1, 8))

    if ai_summary:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Paragraph(ai_summary.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 8))

    # Top issues table
    df = df.copy()
    df["issue_count"] = df["issues"].apply(len)
    top = df.sort_values(["issue_count", "status"], ascending=[False, True]).head(15)
    table_data = [["URL", "Status", "Issues"]]
    for _, r in top.iterrows():
        table_data.append([r["url"], str(r["status"]), "; ".join(r["issues"]) or "None"])
    tbl = Table(table_data, colWidths=[95*mm, 20*mm, 60*mm])
    tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tbl)

    story.append(Spacer(1, 8))
    story.append(Paragraph("Schema Usage (JSON‑LD types detected)", styles["Heading2"]))
    # quick stats
    flat_types = []
    for types in df["jsonld_types"].tolist():
        flat_types.extend(types if isinstance(types, list) else [])
    type_counts = pd.Series(flat_types).value_counts().head(10) if flat_types else pd.Series([])
    if len(type_counts):
        table_data = [["Type", "Count"]] + [[t, int(c)] for t, c in type_counts.items()]
        tbl2 = Table(table_data, colWidths=[60*mm, 30*mm])
        tbl2.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 0.5, colors.black),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]))
        story.append(tbl2)

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ---------------------- OpenAI upgraded scoring ----------------------

def ai_prompt_coverage_upgrade(site: str, samples: List[Tuple[str, str]], api_key: str) -> str:
    if not OPENAI_AVAILABLE:
        return "(OpenAI not installed)"
    try:
        openai.api_key = api_key
        prompts = []
        for url, text in samples[:12]:  # cap to reduce cost
            prompts.append(f"URL: {url}\nCONTENT SAMPLE:\n{text[:1200]}\n---")
        content = (
            "You are an AI search quality rater. From the samples below, infer 10 high-intent user prompts the site likely answers,\n"
            "and estimate overall coverage strength (low/med/high). Return: bullet list of prompts + one-sentence verdict.\n\n" + "\n".join(prompts)
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.4, max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI scoring unavailable: {e})"

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="AI SEO Site Auditor", page_icon="🤖", layout="wide")
st.title("🤖 AI SEO Site Auditor – Whole‑Site Crawl & Report")

with st.sidebar:
    st.header("Settings")
    start_url = st.text_input("Start URL", placeholder="https://example.com")
    max_pages = st.slider("Max pages", 10, 300, 60, step=10)
    max_depth = st.slider("Max crawl depth", 1, 6, 3)
    allow_ai = st.checkbox("Upgrade prompt coverage with OpenAI (optional)")
    openai_key = st.text_input("OpenAI API Key", type="password") if allow_ai else ""
    run_btn = st.button("🚀 Run Audit")

if run_btn:
    if not start_url:
        st.error("Please enter a start URL.")
        st.stop()

    with st.status("Crawling site…", expanded=False) as status:
        pages = crawl_site(start_url, max_pages=max_pages, max_depth=max_depth)
        status.update(label=f"Crawled {len(pages)} pages. Analyzing…")

    results: List[Dict] = []
    prog = st.progress(0.0)

    for i, p in enumerate(pages):
        results.append(analyze_page(p))
        prog.progress((i + 1) / max(1, len(pages)))

    df = pd.DataFrame(results)

    # Site‑wide checks
    sitemap_ok = check_sitemap(start_url)
    df["issue_count"] = df["issues"].apply(len)

    st.success(f"Analyzed {len(df)} pages from {urlparse(start_url).netloc}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pages crawled", len(df))
    k2.metric("Avg title length", int(df["title_len"].replace(0, pd.NA).mean(skipna=True) or 0))
    k3.metric("Pages with JSON‑LD", int((df["jsonld_types"].str.len() > 0).sum()))
    k4.metric("Sitemap found", "Yes" if sitemap_ok else "No")

    st.subheader("Page‑level details")
    st.dataframe(
        df.sort_values(["issue_count", "status"], ascending=[False, True])[[
            "url","status","title_len","meta_description_len","h1_count","word_count",
            "int_links","ext_links","images","images_missing_alt","jsonld_types",
            "prompts_found","prompts_covered","issues"
        ]], use_container_width=True, hide_index=True
    )

    # Prompt coverage (site‑level)
    st.subheader("🧠 Prompt coverage (heuristic)")
    total_prompts = int(df["prompts_found"].sum())
    total_covered = int(df["prompts_covered"].sum())
    st.write(f"Estimated prompts found across pages: **{total_prompts}**")
    st.write(f"Estimated prompts covered (lexical match): **{total_covered}**")

    ai_summary = ""
    if allow_ai and openai_key:
        with st.spinner("Generating AI prompt coverage upgrade…"):
            # build small samples from top pages by word count
            samples: List[Tuple[str, str]] = []
            for _, r in df.sort_values("word_count", ascending=False).head(8).iterrows():
                try:
                    resp = requests.get(r["url"], headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
                    soup = BeautifulSoup(resp.text, "lxml")
                    text = clean_text(soup)
                    samples.append((r["url"], text))
                except Exception:
                    continue
            ai_summary = ai_prompt_coverage_upgrade(urlparse(start_url).netloc, samples, openai_key)
        st.markdown("### AI Prompt Coverage – Summary")
        st.write(ai_summary)
    elif allow_ai:
        st.info("Enter an OpenAI API key to enable upgraded prompt coverage.")

    # ---------------- Lead form + PDF export ----------------
    st.subheader("Download client‑ready PDF")
    with st.form("lead_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Your name")
        email = c2.text_input("Business email")
        company = c3.text_input("Company")
        agree = st.checkbox("I agree to be contacted about this audit.")
        submitted = st.form_submit_button("Generate PDF")

    pdf_bytes: bytes | None = None
    if submitted:
        if not (name and email and company and agree):
            st.error("Please fill in all fields and agree to proceed.")
        else:
            with st.spinner("Building PDF…"):
                pdf_bytes = build_pdf(urlparse(start_url).netloc, df, ai_summary)
            st.success("PDF is ready! Use the button below to download.")

    if pdf_bytes:
        st.download_button(
            "⬇️ Download AI SEO Report (PDF)", data=pdf_bytes,
            file_name="ai_seo_audit.pdf", mime="application/pdf"
        )

else:
    st.markdown(
        """
        **How to use**
        1) Enter a Start URL in the left sidebar.
        2) Click **Run Audit** to crawl the site (same domain). We respect robots.txt and use a polite crawl rate.
        3) Review page‑level issues, JSON‑LD schema usage, and prompt coverage estimates.
        4) (Optional) Provide an OpenAI key to get an upgraded AI prompt coverage summary.
        5) Fill the form to enable the **PDF report download** for sharing with leads.

        **What counts as AI‑SEO readiness**
        - Clean metadata (title/meta description length), H1 presence, canonicals, robots directives OK.
        - JSON‑LD schema (FAQ, HowTo, Product, Organization, WebSite SearchAction, etc.).
        - Content depth and internal linking.
        - Sitemap availability.

        **About "prompts you appear in"**
        We estimate the number of likely user prompts by extracting questions and headings from your pages, then
        checking how many of those your titles/meta/H1s plausibly answer (lexical match). Enable the OpenAI
        option to get a smarter, model‑rated summary of high‑intent prompts.
        """
    )
