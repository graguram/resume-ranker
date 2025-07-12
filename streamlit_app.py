import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from src.utils import clean_text
from src.extract_text import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ‑‑‑ Streamlit page setup ‑‑‑
st.set_page_config(page_title="Resume Ranker", layout="centered")
st.title("📄 Resume Ranker")
st.markdown(
    "Upload one or more PDF résumés, paste the job description, and I’ll rank the candidates by relevance."
)

# ------------------ Inputs ------------------
uploaded_files = st.file_uploader(
    "Upload PDF resumes", type=["pdf"], accept_multiple_files=True
)
jd_text = st.text_area("Paste Job Description", height=150)

# ------------------ Main action ------------------
if st.button("🔍 Rank Resumes"):

    # basic validation
    if not uploaded_files or not jd_text.strip():
        st.warning("Please upload at least one résumé **and** enter a job description.")
        st.stop()

    # ---------- Step 1: Extract text from every PDF ----------
    resume_texts: dict[str, str] = {}
    for up_file in uploaded_files:
        if up_file.size == 0:
            st.warning(f"⚠️  {up_file.name} appears to be empty – skipped.")
            continue

        up_file.seek(0)  # 🔑 reset pointer in the in‑memory buffer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up_file.read())

        text = extract_text_from_pdf(Path(tmp.name))
        resume_texts[up_file.name] = text

    if not resume_texts:
        st.error("No valid PDFs were processed. Please try again.")
        st.stop()

    # ---------- Step 2: Build corpus ----------
    corpus = [clean_text(jd_text)]                 # index 0 → job description
    filenames = []
    for name, txt in resume_texts.items():
        filenames.append(name)
        corpus.append(clean_text(txt))

    # ---------- Step 3: TF‑IDF + cosine similarity ----------
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)              # shape: (n_docs, n_terms)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # ---------- Step 4: Results dataframe ----------
    df = (
        pd.DataFrame(
            {"Resume": filenames, "Similarity Score": [round(s, 3) for s in scores]}
        )
        .sort_values("Similarity Score", ascending=False)
        .reset_index(drop=True)
    )

    # ---------- Display & download ----------
    st.subheader("📊 Ranked Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf‑8")
    st.download_button(
        "Download CSV", csv, file_name="ranked_results.csv", mime="text/csv"
    )
