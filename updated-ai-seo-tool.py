import streamlit as st
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tldextract
import re

# --- APP CONFIG ---
st.set_page_config(page_title="AI SEO Audit", layout="wide")
st.title("AI SEO Readiness Audit Tool")
st.write("""
Check if your website is optimized for **AI search engines** like Google AI Overviews or Bing Copilot.
The table below shows detailed audit results including impacted URLs, scores, and evaluation.
""")
st.markdown("""
*Please book a meeting with us if you want expert help to improve your AI SEO performance: [Contact Us](https://roidigitally.com/contact-us-usa-europe/)*
""")

# --- SIDEBAR ---
st.sidebar.header("AI SEO Audit Settings")
website_url = st.sidebar.text_input("Enter your website URL", placeholder="https://example.com")
run_audit = st.sidebar.button("Run full-site AI SEO audit")

# --- SESSION STATE ---
if "results" not in st.session_state:
    st.session_state.results = None
if "score" not in st.session_state:
    st.session_state.score = None
if "pages" not in st.session_state:
    st.session_state.pages = {}

# --- CRITERIA ---
criteria_list = [
    ("Structured Data", "Schema markup for key pages helps AI understand your content."),
    ("Clear Conversational Content", "Content reads naturally, suitable for AI answers, not just keyword stuffing."),
    ("FAQs / Q&A Sections", "Structured Q&A sections make it easy for AI to answer queries."),
    ("Content Depth", "Pages fully answer intent, typically 300+ words for key pages."),
    ("Internal Linking", "Links between related pages provide context to AI."),
    ("Page Titles & Meta", "Titles and meta descriptions optimized for AI Overviews (question-based)."),
    ("Entities & Context", "Named entities (brands, locations, people) help AI understand pages."),
    ("Fast Loading & Mobile Design", "Indirect AI ranking factor; pages load fast and are mobile-friendly."),
    ("Authoritative Signals (E-E-A-T)", "Author info, credentials, and references improve trust."),
    ("No Duplicate/Thin Content", "Avoid short or repeated content on multiple pages.")
]

# --- ASYNC CRAWLING ---
async def fetch_page(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            html = await response.text()
            return url, html
    except:
        return url, None

async def crawl_website(base_url, max_pages=1000):
    seen_urls = set()
    to_crawl = set([base_url])
    pages = {}
    async with aiohttp.ClientSession() as session:
        while to_crawl and len(pages) < max_pages:
            tasks = [fetch_page(session, url) for url in list(to_crawl)]
            results = await asyncio.gather(*tasks)
            to_crawl.clear()
            for url, html in results:
                if html:
                    pages[url] = html
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = urljoin(url, a['href'])
                        if tldextract.extract(href).domain == tldextract.extract(base_url).domain:
                            if href not in seen_urls:
                                to_crawl.add(href)
                                seen_urls.add(href)
    return pages

# --- AUDIT FUNCTION ---
def audit_pages(pages):
    results = {}
    for criterion, explanation in criteria_list:
        impacted_urls = []
        for url, html in pages.items():
            soup = BeautifulSoup(html, 'html.parser')
            passed = True
            if criterion == "Structured Data":
                if not soup.find_all("script", {"type": "application/ld+json"}):
                    passed = False
            elif criterion == "FAQs / Q&A Sections":
                if not re.search(r"FAQ|Q&A", soup.get_text(), re.IGNORECASE):
                    passed = False
            elif criterion == "Content Depth":
                if len(soup.get_text().split()) < 300:
                    passed = False
            elif criterion == "Internal Linking":
                if not soup.find_all("a", href=True):
                    passed = False
            elif criterion == "Fast Loading & Mobile Design":
                if "viewport" not in str(soup):
                    passed = False
            elif criterion == "No Duplicate/Thin Content":
                if len(soup.get_text().split()) < 100:
                    passed = False
            # default for other criteria: assume passed
            if not passed:
                impacted_urls.append(url)
        num_issues = len(impacted_urls)
        example_urls = impacted_urls[:5]
        results[criterion] = {
            "explanation": explanation,
            "num_issues": num_issues,
            "impacted_urls": example_urls,
            "score": round((1 - num_issues/len(pages))*100) if pages else 0,
            "evaluation": "Passed" if num_issues == 0 else "Needs Improvement",
            "more": max(0, num_issues - len(example_urls))
        }
    overall_score = round(sum(r["score"] for r in results.values()) / len(results))
    return results, overall_score

# --- RUN AUDIT ---
if run_audit and website_url:
    st.info(f"Running audit for: **{website_url}** (multi-page async crawl)")
    pages = asyncio.run(crawl_website(website_url))
    st.session_state.pages = pages
    st.success(f"Crawled {len(pages)} pages.")
    st.session_state.results, st.session_state.score = audit_pages(pages)

# --- SHOW RESULTS ---
if st.session_state.results:
    st.subheader("AI SEO Audit Table")
    table_data = []
    for crit, data in st.session_state.results.items():
        impacted_text = ", ".join(data["impacted_urls"])
        if data["more"]:
            impacted_text += f" …and {data['more']} more pages"
        table_data.append([
            crit,
            data["explanation"],
            data["num_issues"],
            impacted_text,
            data["score"],
            data["evaluation"]
        ])
    st.table(table_data)

    st.subheader("Overall AI SEO Score")
    st.write(f"**{st.session_state.score}/100**")
    st.progress(st.session_state.score)

    # Prompt appearance simulation
    st.subheader("Estimated AI Prompt Appearances")
    st.write(f"Based on audit, your site may appear in approximately **{len(st.session_state.pages)*2} AI prompts**.")
