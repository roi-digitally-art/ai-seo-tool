import streamlit as st
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tldextract
import re
import pandas as pd
from io import BytesIO
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="AI SEO Audit", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"] {
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
*If you think the tool is not accurate, or want to give feedback, or want expert help to improve your AI SEO performance, please [book a meeting with us](https://roidigitally.com/contact-us-usa-europe/).*
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
    ("FAQs / Q&A Sections", "Structured Q&A sections or FAQ schema make it easy for AI to answer queries."),
    ("Content Depth", "Pages fully answer intent, typically 300+ words for key pages."),
    ("Internal Linking", "Links between related pages provide context to AI."),
    ("Page Titles & Meta", "Titles and meta descriptions optimized for AI Overviews (question-based)."),
    ("Entities & Context", "Named entities (brands, locations, people) help AI understand pages."),
    ("Fast Loading & Mobile Design", "Indirect AI ranking factor; pages load fast and are mobile-friendly."),
    ("Authoritative Signals (E-E-A-T)", "Author info, credentials, and references improve trust."),
    ("No Duplicate/Thin Content", "Avoid short or repeated content on multiple pages.")
]

# --- ASYNC CRAWLING ---
semaphore = asyncio.Semaphore(20)  # concurrency limit

async def fetch_page(session, url):
    async with semaphore:
        try:
            async with session.get(url, timeout=15) as response:
                html = await response.text()
                return url, html
        except:
            return url, None

async def crawl_website(base_url):
    seen_urls = set()
    to_crawl = set([base_url])
    pages = {}
    async with aiohttp.ClientSession() as session:
        progress_bar = st.progress(0)
        total = 1
        while to_crawl:
            batch = list(to_crawl)[:20]
            to_crawl.difference_update(batch)
            tasks = [fetch_page(session, url) for url in batch]
            results = await asyncio.gather(*tasks)
            for url, html in results:
                if html:
                    pages[url] = html
                    seen_urls.add(url.rstrip('/'))
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if tldextract.extract(href).domain == tldextract.extract(base_url).domain:
                            if href not in seen_urls:
                                to_crawl.add(href)
            # Update progress estimate
            progress_bar.progress(min(100, int(len(pages)/total*100)))
            total = max(total, len(pages)+len(to_crawl))
    return pages

# --- AUDIT FUNCTION ---
def audit_pages(pages):
    results = {}
    for criterion, explanation in criteria_list:
        impacted_urls = []
        for url, html in pages.items():
            soup = BeautifulSoup(html, 'html.parser')
            passed = True
            text_content = soup.get_text(separator=' ', strip=True)
            # --- Checks ---
            if criterion == "Structured Data":
                if not soup.find_all("script", {"type": "application/ld+json"}):
                    passed = False
            elif criterion == "FAQs / Q&A Sections":
                faqs = soup.find_all(attrs={"itemtype": "https://schema.org/FAQPage"})
                if not faqs and not re.search(r"FAQ|Q&A", text_content, re.IGNORECASE):
                    passed = False
            elif criterion == "Content Depth":
                if len(text_content.split()) < 300:
                    passed = False
            elif criterion == "Internal Linking":
                if len(soup.find_all("a", href=True)) < 2:
                    passed = False
            elif criterion == "Fast Loading & Mobile Design":
                if "viewport" not in str(soup):
                    passed = False
            elif criterion == "No Duplicate/Thin Content":
                if len(text_content.split()) < 100:
                    passed = False
            if not passed:
                impacted_urls.append(url)
        num_issues = len(impacted_urls)
        example_urls = impacted_urls[:5]
        results[criterion] = {
            "explanation": explanation,
            "num_issues": num_issues,
            "impacted_urls": example_urls,
            "more": max(0, num_issues - len(example_urls)),
            "score": round((1 - num_issues/len(pages))*100) if pages else 0,
            "evaluation": "Passed" if num_issues == 0 else "Needs Improvement"
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
    st.dataframe(df, height=400)  # view-only table

    st.subheader("Overall AI SEO Score")
    st.write(f"**{st.session_state.score}/100**")
    st.progress(st.session_state.score)

    # --- AI Prompt Appearance Simulation (Content-based) ---
st.subheader("Estimated AI Prompt Appearances")
total_prompts = 0
page_details = {}

for url, html in st.session_state.pages.items():
    soup = BeautifulSoup(html, 'html.parser')
    page_score = 0
    details = []
    
    # Structured Data
    if soup.find_all("script", {"type": "application/ld+json"}):
        page_score += 1
        details.append("Structured Data present")
    
    # FAQs / Q&A
    if re.search(r"FAQ|Q&A", soup.get_text(), re.IGNORECASE):
        page_score += 1
        details.append("FAQ/Q&A content")
    
    # Content depth
    if len(soup.get_text().split()) >= 300:
        page_score += 1
        details.append("Sufficient content depth")
    
    # Named entities
    entities = re.findall(r"\b[A-Z][a-z]{2,}\b", soup.get_text())  # simple NER example
    if entities:
        page_score += 1
        details.append(f"{len(entities)} named entities")
    
    total_prompts += page_score
    page_details[url] = {"score": page_score, "details": details}

st.write(f"Based on audit, your website may appear in approximately **{total_prompts} AI prompts** across all pages.")

# Optional: show details per page
for url, data in page_details.items():
    st.markdown(f"- [{url}]({url}): {data['score']} prompts → {', '.join(data['details'])}")

# --- USER FORM FOR PDF ---
st.subheader("Download Detailed PDF Report")
with st.form("download_form"):
    name = st.text_input("Your Name", placeholder="John Doe", max_chars=100)
    email = st.text_input("Your Email", placeholder="john@example.com")
    agree_contact = st.checkbox("I agree to be contacted about this audit", value=False)
    submit_form = st.form_submit_button("Generate PDF Report")

    if submit_form:
        if not name or not email:
            st.error("Please fill in all required fields.")
        elif not agree_contact:
            st.error("You must agree to be contacted to download the report.")
        else:
            # --- GENERATE PDF ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI SEO Audit Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Website: {website_url}", ln=True)
            pdf.cell(0, 8, f"Name: {name}", ln=True)
            pdf.cell(0, 8, f"Email: {email}", ln=True)
            pdf.ln(5)
            pdf.cell(0, 8, f"Overall AI SEO Score: {st.session_state.score}/100", ln=True)
            pdf.cell(0, 8, f"Estimated AI Prompt Appearances: {round(st.session_state.total_prompt_score)}", ln=True)
            pdf.ln(5)
            # Add table
            pdf.set_font("Arial", "B", 12)
            for crit, data in st.session_state.results.items():
                pdf.multi_cell(0, 6, f"Criterion: {crit}")
                pdf.set_font("Arial", "", 11)
                pdf.multi_cell(0, 6, f"Explanation: {data['explanation']}")
                pdf.multi_cell(0, 6, f"# Issues: {data['num_issues']}")
                impacted_text = ", ".join(data["impacted_urls"])
                if data["more"]:
                    impacted_text += f", ...and {data['more']} more pages"
                pdf.multi_cell(0, 6, f"Impacted URLs: {impacted_text}")
                pdf.multi_cell(0, 6, f"Recommendations: {', '.join(data['recommendations'])}")
                pdf.multi_cell(0, 6, f"Score: {data['score']} | Evaluation: {data['evaluation']}")
                pdf.ln(3)
                pdf.set_font("Arial", "B", 12)
            # Output PDF
            pdf_output = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="Download PDF Report",
                data=pdf_output,
                file_name="AI_SEO_Audit_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF report generated successfully!")

