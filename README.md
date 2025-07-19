# 🧠 Resume Ranker (AI-Powered Resume Screening Tool)

This project helps HR professionals or recruiters automatically **rank resumes** based on how well they match a **job description**, using AI techniques like **TF-IDF** and **cosine similarity**. It also comes with a **Streamlit web interface** for easy interaction.

## 🚀 Features

- ✅ Upload multiple PDF resumes
- ✅ Paste a job description
- ✅ Automatically rank resumes based on relevance
- ✅ View similarity scores
- ✅ Download ranked results as CSV
- ✅ Clean, user-friendly web UI with Streamlit

## 🛠 Technologies Used

- Python 3
- Streamlit
- scikit-learn
- PyMuPDF
- NLTK
- Pandas

## 📦 Installation

1. Clone this repo or download the ZIP  
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate       # for Windows
# or
source venv/bin/activate      # for macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

4. Run the app:

```bash
streamlit run streamlit_app.py
```

## 📂 Folder Structure

```
resume-ranker/
├── data/
│   ├── resumes/               # Upload resumes here
│   └── job_description.txt    # Paste JD here
├── output/                    # Ranked results CSV
├── src/                       # Logic for scoring & cleaning
├── streamlit_app.py           # Web app entry point
├── app.py                     # CLI version
├── requirements.txt
└── README.md
```

## ✨ Author

Made with ❤️ by G Raguram  
Feel free to use, modify, or contribute!
