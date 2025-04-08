import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from fpdf import FPDF
import re
import openai
import json
from openai import OpenAI

# === Step 1: User Inputs via CLI ===
print("--- Survey Feedback Analysis CLI ---")
survey_name = input("Enter survey name: ")
survey_question = input("Enter the survey question: ")
api_key = input("Enter your OpenAI API key: ")
n_clusters = int(input("Enter number of clusters to group responses (e.g. 4): "))

print("\nPaste your responses below. Type END on a new line to finish.")
responses = []
while True:
    line = input()
    if line.strip().upper() == "END":
        break
    if line.strip():
        responses.append(line.strip())

# === OpenAI client ===
client = OpenAI(api_key=api_key)

# === Step 2: Clustering ===
def cluster_responses(responses, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(responses)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return pd.DataFrame({'Response': responses, 'Cluster': clusters})

# === Step 3: Sentiment Analysis ===
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# === Step 4: Keyword Extraction ===
def extract_keywords(df, max_keywords=15):
    cluster_keywords = {}
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_texts = df[df['Cluster'] == cluster_id]['Response']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 2))
        X_cluster = vectorizer.fit_transform(cluster_texts)
        scores = np.asarray(X_cluster.sum(axis=0)).flatten()
        keywords = list(zip(vectorizer.get_feature_names_out(), scores))
        sorted_keywords = sorted(keywords, key=lambda x: -x[1])
        cluster_keywords[cluster_id] = sorted_keywords[:max_keywords]
    return cluster_keywords

# === Step 5: GPT-Powered Summary ===
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

# === Step 6: PDF Report ===
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
        self.multi_cell(0, 10, clean(f"Insights:\n{insight}\n\nAction Suggestions:\n- " + "\n- ".join(actions)))
        if keywords:
            keyword_list = ', '.join([kw for kw, _ in keywords])
            self.set_font("Arial", "I", 10)
            self.multi_cell(0, 10, f"Top Keywords: {keyword_list}")
        self.ln(5)

# === Step 7: Execute ===
print("\nAnalyzing responses...")
df = cluster_responses(responses, n_clusters=n_clusters)
df['Sentiment'] = df['Response'].apply(analyze_sentiment)
keywords_by_cluster = extract_keywords(df)

summary_per_cluster = {}
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_responses = df[df['Cluster'] == cluster_id]['Response'].tolist()
    result = generate_cluster_summary(cluster_id, cluster_responses)
    summary_per_cluster[cluster_id] = {
        "title": f"Cluster {cluster_id} - {result['title']}",
        "insight": result['insight'],
        "actions": result['actions']
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

pdf_filename = "Customer_Feedback_Analysis_Report_CLI.pdf"
pdf.output(pdf_filename)
print(f"\n✅ PDF report saved as '{pdf_filename}'")
