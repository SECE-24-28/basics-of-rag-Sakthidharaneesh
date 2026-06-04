# =====================================================
# COMPLETE RAG SYSTEM (GPU SAFE)
# =====================================================

# ----------- FORCE CPU ------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.cuda.is_available = lambda: False

# ----------- IMPORTS ------------
import uuid
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# =====================================================
# 1. CONFIGURE GEMINI
# =====================================================

genai.configure(api_key="")

# Change to gemini-2.5-flash if available
model = genai.GenerativeModel("gemini-2.5-flash")


# =====================================================
# 2. PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


# =====================================================
# 3. CHUNKING WITH OVERLAP
# =====================================================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# =====================================================
# 4. LOAD EMBEDDING MODEL (CPU ONLY)
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# =====================================================
# 5. CREATE CHROMADB COLLECTION
# =====================================================

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="resume_collection"
)


# =====================================================
# 6. STORE EMBEDDINGS IN CHROMADB
# =====================================================

def store_chunks(chunks):
    for chunk in chunks:
        embedding = embedding_model.encode(
            chunk,
            convert_to_numpy=True
        ).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())]
        )


# =====================================================
# 7. RETRIEVE RELEVANT CONTEXT
# =====================================================

def retrieve_context(query, top_k=3):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# =====================================================
# 8. GENERATE ANSWER USING GEMINI
# =====================================================

def answer_question(query):

    context_chunks = retrieve_context(query)
    context = "\n".join(context_chunks)

    prompt = f"""
You are a resume assistant.

Answer ONLY using the resume content below.
If the answer is not found in the resume,
reply exactly:
"Not mentioned in the resume."

Resume:
{context}

Question:
{query}

Answer:
"""

    response = model.generate_content(prompt)
    return response.text


# =====================================================
# 9. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    pdf_path = ""  # Put your resume file in same folder

    print("📄 Extracting text from resume...")
    resume_text = extract_text_from_pdf(pdf_path)

    print("✂️ Chunking text with overlap...")
    chunks = chunk_text(resume_text)

    print("🧠 Generating embeddings & storing in ChromaDB...")
    store_chunks(chunks)

    print("\n✅ RAG System Ready!")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Ask about the resume: ")

        if query.lower() == "exit":
            break

        answer = answer_question(query)

        print("\n💬 Answer:\n")
        print(answer)
        print("\n-----------------------------------------\n")