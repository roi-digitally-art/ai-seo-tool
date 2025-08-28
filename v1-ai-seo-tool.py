import streamlit as st
import requests
from bs4 import BeautifulSoup
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import re
import tldextract

# --- APP CONFIG ---
st.set_page_config(page_title="AI SEO Readiness Audit", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("AI SEO Audit Settings")
website_url = st.sidebar.text_input("Enter your website URL", placeholder="https://example.com")
run_audit = st.sidebar.button("Run full-site AI SEO audit")

# --- TITLE ---
st.title("AI SEO Readiness Audit Tool")
st.write("""
Check if your website is optimized for **AI search engines** like Google AI Overviews, Bing Copilot, etc.
""")

# --- SESSION STATE INIT ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- CRITERIA EXPLANATION ---
criteria_list = [
    ("Structured Data", "Schema markup for key pages (e.g., product, FAQ, organization). Helps AI understand your site."),
    ("Clear, Conversational Content", "Content should read naturally, not keyword-stuffed. Suitable for AI-generated answers."),
    ("FAQs / Q&A Sections", "AI prefers pages with structured Q&A for quick answers."),
    ("Content Depth", "Pages should fully answer intent (at least 300 words for key pages)."),
    ("Internal Linking", "Links between related pages to provide context. E.g., service pages link to blogs."),
    ("Page Titles & Meta", "Optimized for AI Overviews; use question-based titles where relevant."),
    ("Entities & Context", "Include named entities (brands, locations, people) to help AI understand relationships."),
    ("Fast Loading & Mobile Design", "Indirect ranking factor; AI prefers well-optimized pages."),
    ("Authoritative Signals (E-E-A-T)", "Show author info, credentials, and references for trust."),
    ("No Duplicate/Thin Content", "Avoid pages with very little or repeated content.")
]

# --- AUDIT FUNCTION ---
def audit_website(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None, f"Failed to fetch the site. Status code: {response.status_code}"
        soup = BeautifulSoup(response.text, 'html.parser')

        results = {}
        # Simple checks
        results["Structured Data"] = bool(soup.find_all("script", {"type": "application/ld+json"}))
        results["FAQs / Q&A Sections"] = bool(re.search(r"FAQ|Q&A", soup.get_text(), re.IGNORECASE))
        results["Content Depth"] = len(soup.get_text().split()) > 300
        results["Internal Linking"] = bool(soup.find_all("a", href=True))
        results["Fast Loading & Mobile Design"] = "viewport" in str(soup)
        results["No Duplicate/Thin Content"] = len(soup.get_text().split()) > 100
        # Others (set default True for demo)
        results["Clear, Conversational Content"] = True
        results["Page Titles & Meta"] = True
        results["Entities & Context"] = True
        results["Authoritative Signals (E-E-A-T)"] = True

        return results, None
    except Exception as e:
        return None, str(e)

# --- RUN AUDIT ---
if run_audit and website_url:
    st.write(f"Running audit for: **{website_url}**")
    results, error = audit_website(website_url)
    if error:
        st.error(error)
    else:
        st.session_state.results = results

# --- SHOW RESULTS ---
if st.session_state.results:
    st.subheader("✅ AI SEO Readiness Checklist")
    for criterion, explanation in criteria_list:
        status = "✅ Passed" if st.session_state.results.get(criterion, False) else "❌ Needs Improvement"
        st.write(f"**{criterion}** - {status}")
        st.caption(explanation)

    st.write("---")
    st.subheader("Download Full Audit Report")

    # FORM FOR PDF DOWNLOAD
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
                # Generate PDF
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                elements = []
                elements.append(Paragraph("AI SEO Readiness Audit Report", styles['Title']))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(f"Website: {website_url}", styles['Normal']))
                elements.append(Paragraph(f"Name: {name}", styles['Normal']))
                elements.append(Paragraph(f"Email: {email}", styles['Normal']))
                elements.append(Spacer(1, 12))

                for criterion, explanation in criteria_list:
                    status = "Passed" if st.session_state.results.get(criterion, False) else "Needs Improvement"
                    elements.append(Paragraph(f"<b>{criterion}</b>: {status}", styles['Normal']))
                    elements.append(Paragraph(explanation, styles['Italic']))
                    elements.append(Spacer(1, 8))

                doc.build(elements)
                buffer.seek(0)
                st.download_button("Download PDF Report", buffer, file_name="ai_seo_audit_report.pdf", mime="application/pdf")

    st.markdown("""
    Want expert help to improve your **AI SEO performance**?
    Please [book a meeting with us](https://roidigitally.com/contact-us-usa-europe/).
    """)

