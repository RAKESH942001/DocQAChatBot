# vector_db_utils.py

import os
import pdfplumber
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def extract_text_from_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                yield page_num, text

def process_pdf(pdf_path: str, chunk_size: int = 1800, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    for page_num, page_text in extract_text_from_pdf(pdf_path):
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                }
            )
            documents.append(doc)

    print(f"✅ Chunks from {pdf_path}: {len(documents)}")
    return documents

def process_multiple_pdfs(pdf_paths: List[str]) -> List[Document]:
    all_documents = []
    for pdf_path in pdf_paths:
        all_documents.extend(process_pdf(pdf_path))
    return all_documents

def store_documents_to_chroma(
    documents: List[Document],
    vector_db_path: str,
    model_name: str = "nomic-embed-text"
):
    embeddings = OllamaEmbeddings(model=model_name, show_progress=True)

    if os.path.exists(vector_db_path):
        print("📁 Existing DB found, updating...")
        vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        vector_db.add_documents(documents)
    else:
        print("🆕 Creating new Chroma DB...")
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vector_db_path
        )

    vector_db.persist()
    print(f"✅ Vector DB saved at: {vector_db_path}")

def build_vector_db_from_pdfs(pdf_paths: List[str], vector_db_path: str):
    documents = process_multiple_pdfs(pdf_paths)
    store_documents_to_chroma(documents, vector_db_path)
