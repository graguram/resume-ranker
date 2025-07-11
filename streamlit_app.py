
import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from src.utils import clean_text
from src.extract_text import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Ranker", layout="centered")

st.title("📄 Resume Ranker")

st.markdown("Upload resumes and paste a job description to rank candidates by relevance.")

# Upload resumes
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Job description input
jd_text = st.text_area("Paste Job Description", height=150)

# When button is clicked
if st.button("🔍 Rank Resumes"):

    if not uploaded_files or not jd_text.strip():
        st.warning("Please upload at least one resume and enter a job description.")
        st.stop()

    resume_texts = {}
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)
            text = extract_text_from_pdf(tmp_path)
            resume_texts[file.name] = text

    # Clean and prepare corpus
    corpus = [clean_text(jd_text)]
    filenames = []

    for name, text in resume_texts.items():
        filenames.append(name)
        corpus.append(clean_text(text))

    # TF-IDF + Cosine Similarity
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    df = pd.DataFrame({
        "Resume": filenames,
        "Similarity Score": [round(s, 3) for s in scores]
    }).sort_values("Similarity Score", ascending=False)

    st.subheader("📊 Ranked Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="ranked_results.csv", mime="text/csv")
