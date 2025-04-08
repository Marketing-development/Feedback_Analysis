
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from fpdf import FPDF
import re
import openai
import os
import json
from openai import OpenAI

# Create OpenAI client using API key
client = OpenAI(api_key="sk-proj-*****************************")

# === STEP 1: Define Survey Name, Question and Load Responses ===
#Zagotovka for future Extractions from CSV:

'''
try:
    df_csv = pd.read_csv("responses.csv")  # Ensure this file exists
    responses = df_csv['Response'].dropna().tolist()
except Exception as e:
    print("⚠️ Failed to load CSV. Using hardcoded fallback responses.")
'''

survey_name = "CLTC Membership Feedback - April 2025"
survey_question = "What could we do better or do less of or stop doing?"
responses = [
    "Can’t think of any improvements that need to be implemented",
    "N/A",
    "Increase relevancy of the CLTC course to the life insurance market. Refresh the tools available to CLTC grads",
    "Nothing comes to mind",
    "One of the Study Groups had someone that updated us on legislation, but most of the time there is not a clear focus of support for the study group",
    "na",
    "Can't think of anything",
    "More awareness",
    "Providing My LTC training that is due every two years to keep my license",
    "N/A",
    "Nothing comes to mind",
    "Nothing comes to mind",
    ".",
    "No comment",
    "Being invisible",
    "Initial onboarding is complicated. Customer service at the onset was very ambiguous",
    "Don't know",
    "I'd like to see graphics and posts that I can share on social media",
    "Basics",
    "Better webinar instructions, connections, and earlier notification for planning. Easy access to materials",
    "Nothing that I can think of",
    "Charge less. Especially for long-time customers",
    "Become 'the authority' in media. Provide plug-and-play content CLTC members can use in conversations",
    "I don't have time for regional study groups, but they may be valuable to others",
    "Offer student the option to turn off narration",
    "Fighting Solutions during LTC process. There are impactful stories not covered in the curriculum",
    "More consumer and industry PR",
    "Much of what's offered is pretty basic",
    "NA",
    "See prior response",
    "Not sure",
    "Costs are high",
    "Communication isn't great. Didn't know about the Digest or newsletter. Just started getting emails recently",
    "NOTHING",
    "Consider short-form video/audio versions of Digest articles",
    "na",
    "Overwhelming amount of education; unsure what to focus on",
    "CE credit for webinars",
    "No specifics come to mind",
    "Send out the CLTC ID card every year on renewal",
    "Provide downloadable, printable client materials branded with CLTC logo",
    "Not sure",
    "Nothing",
    "Keep doing surveys. Make calls to graduates. I'm not receiving emails or calls",
    "N/A",
    "Not sure",
    "Stop trying to charge for everything",
    "Communication of available resources is lacking. I didn’t recognize most of the list provided",
    "More engaging activities",
    "x",
    "Have a person read CE content instead of a computer",
    "Everything is fine",
    "No comment",
    "Nothing",
    "Annual fee is a pain (though likely won’t change)",
    "Nothing",
    "High renewal fee",
    "Not sure",
    "Nothing",
    "Split 16-hour online classes over 3–4 days. Eight-hour sessions are too much",
    "Offer CE credits",
    "This is the first correspondence from you in years",
    "N/a",
    "More training",
    "Acknowledge LTC is a tough sell. Be more realistic about client hesitations and market barriers",
    "n/a",
    "Stop sending emails about classes already taken",
    "No comment",
    "Nothing",
    "There might be too much offered. Streamline communications to be more concise",
    "Unknown",
    "n/a",
    "Lower cost classes",
    "Nothing",
    "Offer smaller, bite-sized content",
    "Maintain good communication; offer just-in-time learning via video library",
    "Be a resource to the NAIC on LTC",
    "Unsure",
    "I don’t think there is anything to change. Keep doing what you’re doing",
    "n/a",
    "Offer annual updates for virtual and hardcopy files. Provide low-cost refresher course for certificants",
    "It was strange to offer CE on LTC classes right after I passed my test",
    "Do more to make members aware",
    "Continue to offer white papers and real client stories on LTC planning",
    "Decrease renewal cost",
    "I don’t really know all the membership benefits",
    "Nothing",
    "Reduce fees for the designation. More education shouldn’t mean more fees",
    "Not sure",
    "Nothing comes to mind, what is offered is just right",
    "More charts",
    "N/A",
    "Keep in better touch",
    "I can't think of anything",
    "Nothing :)",
    "Everything",
    "Nothing",
    "Highlight what material will be tested",
    "Stop sending training invitations to people who already completed CLTC",
    "Offer 30-minute webinars"
]

# === STEP 2: Clustering Responses ===
def cluster_responses(responses, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(responses)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    df = pd.DataFrame({'Response': responses, 'Cluster': clusters})
    return df

# === STEP 3: Sentiment Analysis ===
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# === STEP 4: Extract Top Keywords per Cluster ===
def extract_keywords(df, max_keywords=15):
    cluster_keywords = {}

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_texts = df[df['Cluster'] == cluster_id]['Response']

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2)
        )
        X_cluster = vectorizer.fit_transform(cluster_texts)
        scores = np.asarray(X_cluster.sum(axis=0)).flatten()
        keywords = list(zip(vectorizer.get_feature_names_out(), scores))
        sorted_keywords = sorted(keywords, key=lambda x: -x[1])
        cluster_keywords[cluster_id] = sorted_keywords[:max_keywords]

    return cluster_keywords

# === STEP 5: GPT-Powered Cluster Summary Generation ===
def generate_cluster_summary(cluster_id, responses):
    prompt = f"""
You are a senior CX analyst.

Survey Topic: {survey_name}
Survey Question: {survey_question}

You are given a group of survey responses. Your job is to:
- Name this cluster with a short, professional title
- Write a 1–2 sentence insight summarizing the common concern or theme
- Provide 3–5 actionable recommendations for the business

Respond in JSON format with the keys: title, insight, actions

Cluster ID: {cluster_id}
Responses:
{chr(10).join(f"- {r}" for r in responses)}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a customer feedback strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content.strip()

    # Clean up markdown-wrapped JSON
    if content.startswith('```json'):
        content = content.replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing GPT response for cluster {cluster_id}:", e)
        print("Response was:", content)
        return {
            "title": f"Cluster {cluster_id} - Undefined",
            "insight": "Insight could not be generated.",
            "actions": ["No actions available due to error."]
        }

# === STEP 6: PDF Generator ===
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Customer Feedback Analysis Report", ln=True, align="C")
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Survey: {survey_name}", ln=True, align="C")
        self.cell(0, 10, f"Question: {survey_question}", ln=True, align="C")
        self.ln(10)

    def add_cluster_section(self, title, insight, actions, keywords=None):
        def clean(text): return re.sub(r'[^\x00-\x7F]+', '-', text)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, clean(title), ln=True)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, clean(f"Insights:\n{insight}\n\nAction Suggestions:\n{actions}\n"))

        if keywords:
            keyword_list = ', '.join([kw for kw, _ in keywords])
            self.set_font("Arial", "I", 10)
            self.multi_cell(0, 10, f"Top Keywords: {keyword_list}")
        self.ln(5)

# === STEP 7: Run Analysis ===
df = cluster_responses(responses)
df['Sentiment'] = df['Response'].apply(analyze_sentiment)
keywords_by_cluster = extract_keywords(df)

summary_per_cluster = {}

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_responses = df[df['Cluster'] == cluster_id]['Response'].tolist()
    result = generate_cluster_summary(cluster_id, cluster_responses)

    summary_per_cluster[cluster_id] = {
        "title": f"Cluster {cluster_id} - {result['title']}",
        "insight": result['insight'],
        "actions": "- " + "\n- ".join(result['actions'])
    }

# Generate PDF
pdf = PDFReport()
pdf.add_page()

for cluster_id in sorted(summary_per_cluster.keys()):
    cluster = summary_per_cluster[cluster_id]
    top_keywords = keywords_by_cluster.get(cluster_id, [])
    pdf.add_cluster_section(
        title=cluster['title'],
        insight=cluster['insight'],
        actions=cluster['actions'],
        keywords=top_keywords
    )

# Save PDF
print ('')
pdf.output("What_can_we_do_better_02.pdf")
print("PDF report saved as 'What_can_we_do_better_02.pdf'")
print ('')
