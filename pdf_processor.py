import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def extract_text(pdf_path: str) -> list:
    """Extract text from PDF using PyMuPDF"""
    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]

def preprocess_text(text_pages: list, min_word_length=3, custom_denylist=None) -> list:
    """Clean and tokenize text for NLP processing"""
    if custom_denylist is None:
        custom_denylist = {"fig", "eq", "ref", "table", "et", "al", "figure", "section", "tab"}
    
    processed = []
    for doc in text_pages:
        tokens = [
            word.lower()
            for word in word_tokenize(doc)
            if word.isalpha() 
            and len(word) >= min_word_length
            and word.lower() not in custom_denylist
        ]
        if tokens:
            processed.append(tokens)
    return processed