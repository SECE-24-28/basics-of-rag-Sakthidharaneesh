# RAG System for Resumes

This project implements a complete Retrieval-Augmented Generation (RAG) system designed to answer questions about a resume provided in PDF format. The system is built to run on a CPU, making it accessible for users without a dedicated GPU.

## Features

- **PDF Text Extraction**: Extracts text from a PDF file.
- **Text Chunking**: Splits the extracted text into smaller, overlapping chunks.
- **Embeddings**: Generates embeddings for each chunk using the `all-MiniLM-L6-v2` model.
- **Vector Storage**: Stores the text chunks and their corresponding embeddings in a ChromaDB collection.
- **Context Retrieval**: Retrieves the most relevant text chunks from ChromaDB based on a user's query.
- **Answer Generation**: Uses Google's Gemini 1.5 Flash model to generate an answer based on the retrieved context.

## Requirements

- Python 3.x
- `torch`
- `pdfplumber`
- `chromadb`
- `sentence-transformers`
- `google-generativeai`

You can install the necessary packages using pip:

```bash
pip install torch pdfplumber chromadb sentence-transformers google-generativeai
```

## Usage

1.  **Set up your API Key**: In `rag.py`, replace `""` in `genai.configure(api_key="")` with your Google Gemini API key.
2.  **Place your Resume**: Put your resume in PDF format in the same folder as `rag.py`.
3.  **Update the PDF Path**: In `rag.py`, update the `pdf_path` variable with the name of your resume file.
4.  **Run the script**:

    ```bash
    python rag.py
    ```

5.  **Ask Questions**: Once the system is ready, you can ask questions about the resume in the terminal. Type `exit` to stop the program.

## How it Works

1.  **Text Extraction**: The script starts by extracting all the text from the provided PDF resume.
2.  **Chunking**: The extracted text is then broken down into smaller chunks of 500 characters with an overlap of 100 characters between them. This helps in maintaining context.
3.  **Embedding and Storage**: Each chunk is converted into a numerical vector (embedding) using the `SentenceTransformer` model. These embeddings are then stored in a ChromaDB collection.
4.  **Retrieval**: When a user asks a question, the query is also converted into an embedding. ChromaDB then performs a similarity search to find the most relevant text chunks from the stored collection.
5.  **Generation**: The retrieved chunks are passed as context to the Gemini 1.5 Flash model along with the user's query. The model then generates a natural language answer based on the provided context. If the answer is not found in the resume, it will reply with "Not mentioned in the resume."
