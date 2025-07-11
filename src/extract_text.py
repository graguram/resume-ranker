from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: Path) -> str:
    text_blocks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_blocks.append(page.get_text())
    return "\n".join(text_blocks)

def load_resumes(resume_dir: Path) -> dict[str, str]:
    resumes = {}
    for pdf in resume_dir.glob("*.pdf"):
        resumes[pdf.name] = extract_text_from_pdf(pdf)
    return resumes

if __name__ == "__main__":
    sample = load_resumes(Path("../data/resumes"))
    print({k: v[:300] for k, v in sample.items()})
