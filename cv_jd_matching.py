import re
import string
import contractions
import pdfplumber
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += " " + page_text
    return text.strip()

def text_cleaning(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    translator = str.maketrans('', '', string.punctuation)
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{1,3}[-./]?\d{1,3}[-./]?\d{1,4}\b', '', text)
    text = text.translate(translator)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.strip()

# Load model/tokenizer only once
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    return embeddings[0]

def compute_similarity(cv_text, jd_text):
    cv_clean = text_cleaning(cv_text)
    jd_clean = text_cleaning(jd_text)
    cv_emb = get_embedding(cv_clean)
    jd_emb = get_embedding(jd_clean)
    score = cosine_similarity([cv_emb], [jd_emb])[0][0]
    return float(score)