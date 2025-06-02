
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from collections import Counter, deque
import json
import re
from urllib.parse import urljoin, urlparse


#==================REDIRECT TO TXT FILE=========================================================

# Redirect all standard output (what you'd normally see in the console) to report_output.txt

# Automatically write everything your print() calls generate

import sys
sys.stdout = open("report_output.txt", "w", encoding="utf-8")



# === CONFIGURATION ============================================================================

api_key = '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'
MY_WEBSITE = "https://institutedfa.com/"
COMPETITOR_WEBSITE = "https://us.aicpa.org/forthepublic"
CRAWL_DEPTH = 2
MAX_PAGES = 10

if not api_key.startswith("sk-proj-"):
    raise ValueError("Invalid OpenAI API key format.")
openai = OpenAI(api_key=api_key)

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; SEO-Bot/1.0; +https://institutedfa.com/bot)"
}

# === RECURSIVE CRAWLER ===

def crawl_recursive(base_url, max_depth=2, max_pages=10):
    visited = set()
    queue = deque([(base_url, 0)])
    results = []

    base_domain = urlparse(base_url).netloc

    while queue and len(results) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except:
            continue

        results.append(url)

        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup.find_all("a", href=True):
            full_url = urljoin(url, tag["href"])
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain and full_url not in visited:
                queue.append((full_url, depth + 1))

    return results

# === SCRAPER ===

class Website:
    def __init__(self, urls):
        self.urls = urls
        self.title = ""
        self.meta_description = ""
        self.h1_tags = []
        self.text = ""
        self.structured_data = []
        self.keywords = []

        for url in urls:
            self.scrape_url(url)

        self.keywords = self.extract_keywords()

    def scrape_url(self, url):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except:
            return

        soup = BeautifulSoup(response.content, 'html.parser')

        if not self.title and soup.title and soup.title.string:
            self.title = soup.title.string.strip()
        if not self.meta_description:
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                self.meta_description = meta["content"].strip()

        self.h1_tags.extend(h1.get_text(strip=True) for h1 in soup.find_all("h1"))

        if soup.body:
            for tag in soup.body(["script", "style", "img", "input", "noscript", "iframe"]):
                tag.decompose()
            self.text += soup.body.get_text(separator="\n", strip=True) + "\n"

        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                self.structured_data.append(json.loads(tag.string))
            except:
                continue

    def extract_keywords(self, min_len=4, top_n=20):
        words = re.findall(r'\b\w+\b', self.text.lower())
        filtered = [w for w in words if len(w) >= min_len and not w.startswith(('http', 'www'))]
        freq = Counter(filtered)
        return [kw for kw, _ in freq.most_common(top_n)]

# === PROMPT GENERATION ===

system_prompt = (
    "You are a professional website analysis assistant specializing in UX, SEO, and competitor benchmarking.\n"
    "Your job is to generate a markdown summary of a given website. Highlight:\n"
    "- Strengths and weaknesses\n"
    "- Content quality and clarity\n"
    "- Conversion UX issues\n"
    "- SEO and accessibility\n"
    "- Titles, Meta Descriptions, and H1s\n"
    "- Recommendations for improvements\n"
    "- Top keywords for Google Search Ads\n"
    "- Competitor positioning and value proposition clarity\n"
    "- Target audience tone alignment (B2B, B2C, beginner, executive)\n"
    "- Presence and quality of social proof (testimonials, reviews, logos, certifications)\n"
    "- Clarity and effectiveness of Calls-To-Action (CTAs)\n"
    "- Lead generation mechanisms (forms, signup flows, CTAs, free trials)\n"
    "- Technical performance risks (page speed, mobile-first, image loading, JS bloat)\n"
    "- Trust-building signals (HTTPS, about us, contact info, legal disclaimers, authority badges)\n"
    "- Include comparative SEO gaps or missed opportunities against similar domains\n"
    "- Estimate content freshness: are blogs, dates, and formats updated or stale?\n"
    "- Evaluate AI-generated vs. human-written content tone (if detectable)"
)


def user_prompt_for(website: Website):
    return (
        f"Website Title: {website.title}\n"
        f"Meta Description: {website.meta_description}\n"
        f"H1 Tags: {website.h1_tags}\n\n"
        f"Text Content:\n{website.text[:5000]}...\n\n"
        "Please also check pricing details, memberships, and summarize pain points and strengths."
    )

def summarize_website(url):
    urls_to_scrape = crawl_recursive(url, max_depth=CRAWL_DEPTH, max_pages=MAX_PAGES)
    site = Website(urls_to_scrape)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(site)}
    ]
    response = openai.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def compare_summaries(summary_1, summary_2):
    messages = [
        {"role": "system", "content": (
            "You are an expert digital marketing consultant.\n"
            "Compare these two websites based on:\n"
            "- SEO optimization and structure\n"
            "- Content clarity and calls-to-action\n"
            "- Keyword strategy\n"
            "- User experience and design appeal\n"
            "- Trust signals and conversion potential\n"
            "Give an objective, bullet-point comparison and provide final recommendations."
        )},
        {"role": "user", "content": (
            f"Website A Summary:\n{summary_1}\n\n---\n\nWebsite B Summary:\n{summary_2}"
        )}
    ]
    response = openai.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

# === RUN FULL ANALYSIS ===

def run_comparison(my_url, competitor_url):
    print(f"\n🔍 Crawling and analyzing YOUR site: {my_url}")
    my_summary = summarize_website(my_url)
    print("\n=== 🟢 Your Site Summary ===\n")
    print(my_summary)

    print(f"\n🔍 Crawling and analyzing COMPETITOR site: {competitor_url}")
    competitor_summary = summarize_website(competitor_url)
    print("\n=== 🔴 Competitor Site Summary ===\n")
    print(competitor_summary)

    print("\n⚔️ Running final head-to-head comparison...")
    comparison = compare_summaries(my_summary, competitor_summary)
    print("\n=== 🏁 HEAD-TO-HEAD ANALYSIS ===\n")
    print(comparison)

# === START ===
run_comparison(MY_WEBSITE, COMPETITOR_WEBSITE)
