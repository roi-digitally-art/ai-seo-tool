import streamlit as st
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tldextract
import re
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="AI SEO Audit", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Manrope', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("AI SEO Readiness Audit Tool")
st.write("""
Check if your website is optimized for **AI search engines** like Google AI Overviews or Bing Copilot.
The table below shows detailed audit results including impacted URLs, scores, and evaluation.
""")
st.markdown("""
*If you think the tool is not accurate, or want to give us feedback, or want expert help to improve your AI SEO performance, feel free to book a meeting with us: [Contact Us](https://roidigitally.com/contact-us-usa-europe/)*
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

async def crawl_website(base_url, batch_size=10, max_pages=1000):
    seen_urls = set([base_url])
    to_crawl = set([base_url])
    pages = {}
    async with aiohttp.ClientSession() as session:
        while to_crawl and len(pages) < max_pages:
            batch = list(to_crawl)[:batch_size]
            to_crawl.difference_update(batch)
            tasks = [fetch_page(session, url) for url in batch]
            results = await asyncio.gather(*tasks)
            for url, html in results:
                if html:
                    pages[url] = html
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = urljoin(url, a['href'])
                        if tldextract.extract(href).domain == tldextract.extract(base_url).domain:
                            if href not in seen_urls:
                                seen_urls.add(href)
                                to_crawl.add(href)
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
    table_rows = []
    for crit, data in st.session_state.results.items():
        impacted_text = []
        for u in data["impacted_urls"]:
            impacted_text.append(f"[{u}]({u})")
        if data["more"]:
            impacted_text.append(f"...and {data['more']} more pages")
        table_rows.append({
            "Criterion": crit,
            "Explanation": data["explanation"],
            "# Issues": data["num_issues"],
            "Impacted URLs": ", ".join(impacted_text),
            "Score (%)": data["score"],
            "Evaluation": data["evaluation"]
        })
    df = pd.DataFrame(table_rows)
    st.dataframe(df, height=400)

    st.subheader("Overall AI SEO Score")
    st.write(f"**{st.session_state.score}/100**")
    st.progress(st.session_state.score)

    # --- AI Prompt Appearance Simulation ---
    st.subheader("Estimated AI Prompt Appearances")
    total_prompts = len(st.session_state.pages)*2
    st.write(f"Based on audit, your website may appear in approximately **{total_prompts} AI prompts**.")
    st.markdown("""
**Breakdown by page type (estimate):**
- Home page: ~5 prompts  
- Product/Service pages: ~15 prompts  
- Blog/FAQ pages: ~10 prompts  
- Other pages: ~6 prompts  

**Suggestions to increase AI visibility:**
- Add FAQs to key pages  
- Implement structured data/schema on product pages  
- Enrich content with named entities  
- Increase content depth on underperforming pages
""")
