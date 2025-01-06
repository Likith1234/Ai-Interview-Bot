from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
import fitz  # PyMuPDF for PDF processing
import openai
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import hashlib
import pickle

# OpenAI API Key
openai.api_key = "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"

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for shortlisted resumes
shortlisted_resumes = []

# Cache for storing results
cache = {}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text.strip()

# Function to extract skills using GPT-4
def extract_skills_with_gpt(text):
    prompt = (
        f"Extract the primary and secondary skills from the following text:\n\n{text}\n\n"
        "Return the skills as a comma-separated list."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant skilled at extracting information from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # Ensure deterministic output
        max_tokens=200
    )
    skills_text = response["choices"][0]["message"]["content"]
    skills = [skill.strip() for skill in skills_text.split(",") if skill.strip()]
    return skills

# Function to calculate similarity
def calculate_similarity(text1, text2, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs_1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs_2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        embedding_1 = model(**inputs_1).last_hidden_state.mean(dim=1)
        embedding_2 = model(**inputs_2).last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(embedding_1.cpu().numpy(), embedding_2.cpu().numpy())
    return similarity[0][0]

# Function to decide if the resume is selected
def is_resume_selected(jd_text, resume_text, threshold=0.50):
    jd_skills = extract_skills_with_gpt(jd_text)
    resume_skills = extract_skills_with_gpt(resume_text)

    jd_skills_text = " ".join(jd_skills)
    resume_skills_text = " ".join(resume_skills)

    similarity_score = calculate_similarity(jd_skills_text, resume_skills_text)
    return similarity_score >= threshold, similarity_score * 100, resume_skills

# Function to generate a unique hash for caching
def generate_hash(jd_text, resume_text):
    combined_text = jd_text + resume_text
    return hashlib.md5(combined_text.encode()).hexdigest()

# Endpoint to filter resumes
@app.post("/filter-resumes/")
async def filter_resumes(jd_pdf: UploadFile = File(...), resume_pdfs: List[UploadFile] = File(...)):
    global shortlisted_resumes
    shortlisted_resumes = []

    # Save the JD PDF to a temporary file
    jd_pdf_path = f"temp_jd_{jd_pdf.filename}"
    with open(jd_pdf_path, "wb") as f:
        f.write(await jd_pdf.read())

    # Extract text from JD PDF
    jd_text = extract_text_from_pdf(jd_pdf_path)

    # Process each resume
    not_shortlisted = []

    for resume_pdf in resume_pdfs:
        resume_pdf_path = f"temp_resume_{resume_pdf.filename}"
        with open(resume_pdf_path, "wb") as f:
            f.write(await resume_pdf.read())

        resume_text = extract_text_from_pdf(resume_pdf_path)

        # Generate a unique hash for caching
        cache_key = generate_hash(jd_text, resume_text)

        # Check if the result is already in the cache
        if cache_key in cache:
            selected, similarity, resume_skills = cache[cache_key]
        else:
            selected, similarity, resume_skills = is_resume_selected(jd_text, resume_text, threshold=0.50)
            cache[cache_key] = (selected, similarity, resume_skills)

        if selected:
            shortlisted_resumes.append((resume_pdf.filename, similarity, resume_skills))
        else:
            not_shortlisted.append((resume_pdf.filename, similarity))

        # Remove the temporary resume file
        os.remove(resume_pdf_path)

    # Remove the temporary JD file
    os.remove(jd_pdf_path)

    return JSONResponse(content={
        "shortlisted": [(resume[0], resume[1]) for resume in shortlisted_resumes],
        "not_shortlisted": not_shortlisted
    })

# Endpoint to search for skills in shortlisted resumes
@app.post("/search-skills/")
async def search_skills(skill: str = Form(...)):
    matched_resumes = []
    for resume in shortlisted_resumes:
        resume_filename, similarity, resume_skills = resume
        if any(skill.lower() in s.lower() for s in resume_skills):
            matched_resumes.append((resume_filename, similarity))

    return JSONResponse(content={
        "matched_resumes": matched_resumes
    })

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)