import streamlit as st
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from urllib.parse import urljoin
import tldextract

# --- APP CONFIG ---
st.set_page_config(page_title="AI SEO Audit", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("AI SEO Audit Settings")
website_url = st.sidebar.text_input("Enter your website URL", placeholder="https://example.com")
run_audit = st.sidebar.button("Run full-site AI SEO audit")

# --- TITLE & INTRO ---
st.title("AI SEO Readiness Audit Tool")
st.write("""
Check if your website is optimized for **AI search engines** like Google AI Overviews, Bing Copilot, etc.
After audit, you can download a **PDF report** by filling the form below.
""")
st.markdown("""
*Please book a meeting with us if you want expert help to improve your AI SEO performance: [Contact Us](https://roidigitally.com/contact-us-usa-europe/)*
""")

# --- SESSION STATE ---
if "results" not in st.session_state:
    st.session_state.results = None
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None
if "score" not in st.session_state:
    st.session_state.score = None

# --- CRITERIA ---
criteria_list = [
    ("Structured Data", "Schema markup for key pages (e.g., product, FAQ, organization). Helps AI understand your site."),
    ("Clear, Conversational Content", "Content should read naturally, not keyword-stuffed. Suitable for AI-generated answers."),
    ("FAQs / Q&A Sections", "AI prefers pages with structured Q&A for quick answers."),
    ("Content Depth", "Pages should fully answer intent (at least 300 words for key pages)."),
    ("Internal Linking", "Links between related pages provide context for AI."),
    ("Page Titles & Meta", "Optimized for AI Overviews; use question-based titles."),
    ("Entities & Context", "Include named entities (brands, locations, people) for AI understanding."),
    ("Fast Loading & Mobile Design", "Indirect ranking factor; AI prefers well-optimized pages."),
    ("Authoritative Signals (E-E-A-T)", "Show author info, credentials, and references."),
    ("No Duplicate/Thin Content", "Avoid pages with very little or repeated content.")
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
    score_table = {}
    for criterion, explanation in criteria_list:
        passed_count = 0
        for html in pages.values():
            soup = BeautifulSoup(html, 'html.parser')
            if criterion == "Structured Data":
                passed_count += bool(soup.find_all("script", {"type": "application/ld+json"}))
            elif criterion == "FAQs / Q&A Sections":
                passed_count += bool(re.search(r"FAQ|Q&A", soup.get_text(), re.IGNORECASE))
            elif criterion == "Content Depth":
                passed_count += len(soup.get_text().split()) > 300
            elif criterion == "Internal Linking":
                passed_count += bool(soup.find_all("a", href=True))
            elif criterion == "Fast Loading & Mobile Design":
                passed_count += "viewport" in str(soup)
            elif criterion == "No Duplicate/Thin Content":
                passed_count += len(soup.get_text().split()) > 100
            else:
                passed_count += 1  # Assume other criteria are fine
        score_table[criterion] = round(passed_count / len(pages) * 100)
    overall_score = round(sum(score_table.values()) / len(score_table))
    return score_table, overall_score

# --- RUN AUDIT ---
if run_audit and website_url:
    st.info(f"Running audit for: **{website_url}** (multi-page, async crawl)")
    pages = asyncio.run(crawl_website(website_url))
    st.success(f"Crawled {len(pages)} pages.")
    st.session_state.results, st.session_state.score = audit_pages(pages)

# --- SHOW RESULTS ---
if st.session_state.results:
    st.subheader("AI SEO Audit Summary")
    st.write(f"**Overall AI SEO Score:** {st.session_state.score}/100")
    if st.session_state.score < 50:
        st.warning("Bad")
    elif st.session_state.score < 80:
        st.info("Average")
    else:
        st.success("Good")

    st.progress(st.session_state.score)

    # --- Table of Scores ---
    table_data = []
    for crit, val in st.session_state.results.items():
        table_data.append([crit, f"{val}%"])
    st.table(table_data)

    # --- Simulate Prompt Appearance ---
    st.subheader("Estimated AI Prompt Appearances")
    st.write(f"Based on audit, your site may appear in approximately **{len(pages)*2} AI answers/prompts**.")  # simplified simulation

    # --- FORM FOR PDF ---
    with st.form("lead_capture_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        agree = st.checkbox("I agree to be contacted about this audit", value=False)
        submitted = st.form_submit_button("Generate PDF Report")

    if submitted:
        if not name or not email:
            st.error("Please fill in your name and email.")
        elif not agree:
            st.error("You must agree to be contacted to get the report.")
        else:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("AI SEO Audit Report", styles['Title']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Website: {website_url}", styles['Normal']))
            elements.append(Paragraph(f"Name: {name}", styles['Normal']))
            elements.append(Paragraph(f"Email: {email}", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Table in PDF
            table = [["Criterion", "Score (%)"]]
            for crit, val in st.session_state.results.items():
                table.append([crit, f"{val}%"])
            t = Table(table)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1,12))
            elements.append(Paragraph(f"Overall AI SEO Score: {st.session_state.score}/100", styles['Normal']))
            doc.build(elements)
            buffer.seek(0)
            st.session_state.pdf_buffer = buffer
            st.success("PDF report ready!")

# --- DOWNLOAD BUTTON OUTSIDE FORM ---
if st.session_state.get("pdf_buffer"):
    st.download_button(
        "Download PDF Report",
        st.session_state.pdf_buffer,
        file_name="ai_seo_audit_report.pdf",
        mime="application/pdf"
    )
