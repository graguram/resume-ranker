from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import clean_text
from .extract_text import load_resumes

DATA_DIR = Path("data")
OUT_CSV = Path("output/ranked_results.csv")

def rank_resumes():
    resumes = load_resumes(DATA_DIR / "resumes")
    if not resumes:
        print("No resumes found in data/resumes/")
        return

    job_desc = (DATA_DIR / "job_description.txt").read_text(encoding="utf-8")
    jd_clean = clean_text(job_desc)

    filenames = []
    corpus = [jd_clean]  # First is job description

    for name, text in resumes.items():
        filenames.append(name)
        corpus.append(clean_text(text))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    df = pd.DataFrame({"resume": filenames, "similarity": scores}) \
           .sort_values("similarity", ascending=False)

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df)

if __name__ == "__main__":
    rank_resumes()
