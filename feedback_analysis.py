import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import openai
import os
import json
from openai import OpenAI
from fpdf import FPDF
import re
import tempfile

# === Streamlit App ===
st.set_page_config(page_title="Survey Insights Generator", layout="centered")
st.title("📊 Survey Feedback Analyzer with GPT")

# === Survey Metadata Inputs ===
survey_name = st.text_input("Survey Name", "CLTC Membership Feedback - April 2025")
survey_question = st.text_area("Survey Question", "What could we do to increase your likelihood to recommend our certification and educational training?")

# === Input Responses ===
responses_raw = st.text_area("Paste Survey Responses (one per line)")

# === Number of Clusters ===
n_clusters = st.slider("How many feedback clusters to generate?", 2, 10, 3)

# === API Key Input ===
api_key = st.text_input("Enter your OpenAI API key", type="password")

# === Process Button ===
if st.button("Generate Report") and responses_raw and api_key:
    responses = [line.strip() for line in responses_raw.splitlines() if line.strip()]

    client = OpenAI(api_key=api_key)

    def cluster_responses(responses, n_clusters=3):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(responses)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        return pd.DataFrame({'Response': responses, 'Cluster': clusters})

    def analyze_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_keywords(df, max_keywords=10):
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
            st.error(f"Error parsing GPT response for cluster {cluster_id}: {e}")
            return {
                "title": f"Cluster {cluster_id} - Undefined",
                "insight": "Insight could not be generated.",
                "actions": ["No actions available due to error."]
            }

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        st.success("✅ Report generated successfully!")
        with open(tmp_file.name, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="Customer_Feedback_Report.pdf")
