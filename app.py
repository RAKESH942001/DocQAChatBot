from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, traceback, logging, time, hashlib
import numpy as np

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

from prompt import prompt
from llm import llama_client

# ---------- Setup Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- FastAPI Setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Response Models ----------
class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

class ResponseModel(BaseModel):
    answer: str
    retrieved_documents: list[DocumentInfo]

# ---------- In‑Memory Cache for Vector DBs ----------
# key: pdf_md5 → value: (Chroma vector_db, embedding_fn)
vector_db_cache = {}

# ---------- Helper Functions ----------
def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def load_or_build_vector_db(pdf_bytes: bytes, pages, chunk_size=1500, chunk_overlap=200):
    """
    Builds (or loads cached) a Chroma vectorstore for this PDF.
    """
    pdf_hash = md5_bytes(pdf_bytes)
    if pdf_hash in vector_db_cache:
        logging.info(f"[Cache] Reusing vector DB for hash {pdf_hash}")
        return vector_db_cache[pdf_hash]

    # Not cached → split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for p in pages:
        for chunk in splitter.split_text(p.page_content):
            if chunk.strip():
                chunks.append(p.__class__(page_content=chunk, metadata=p.metadata))
    logging.info(f"[VectorDB] Building new DB with {len(chunks)} chunks")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    # cache it
    vector_db_cache[pdf_hash] = (vector_db, embeddings)
    return vector_db, embeddings

def dedupe(docs):
    seen, unique = set(), []
    for d in docs:
        txt = d.page_content.strip()
        if txt and txt not in seen:
            unique.append(d)
            seen.add(txt)
    return unique

def rerank(docs, query, embed_fn, top_k=5):
    qv = embed_fn.embed_query(query)
    scored = [(cosine_similarity(qv, embed_fn.embed_query(d.page_content)), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

# ---------- Main Endpoint ----------
@app.post("/ask", response_model=ResponseModel)
async def ask_question(question: str = Form(...), pdf: UploadFile = File(...)):
    total_start = time.time()
    try:
        # 1. Read PDF & compute hash
        data = await pdf.read()
        step = time.time()

        # 2. Load pages
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        text = "".join(p.page_content for p in pages)
        if not text.strip():
            raise HTTPException(400, "PDF has no extractable text.")
        logging.info(f"Step2 (load & split): {time.time()-step:.2f}s")

        # 3. Vector DB (cached or fresh)
        step = time.time()
        vector_db, embed_fn = load_or_build_vector_db(data, pages)
        logging.info(f"Step3 (vector DB): {time.time()-step:.2f}s")

        # 4. HyDE draft answer
        step = time.time()
        hyde_prompt = f"Generate a concise answer to help with retrieval:\n\nQuestion: {question}"
        hyde_resp = llama_client.invoke(hyde_prompt)
        hyde_answer = getattr(hyde_resp, "content", str(hyde_resp))
        logging.info(f"Step4 (HyDE): {time.time()-step:.2f}s")

        # 5. Fusion Retrieval
        step = time.time()
        base_ret = vector_db.as_retriever(search_kwargs={"k":4})
        retriever = MultiQueryRetriever.from_llm(retriever=base_ret, llm=llama_client)
        docs = dedupe(retriever.get_relevant_documents(hyde_answer) +
                      retriever.get_relevant_documents(question))
        logging.info(f"Step5 (fusion): {time.time()-step:.2f}s")

        # 6. Rerank
        step = time.time()
        top_docs = rerank(docs, question, embed_fn, top_k=5)
        logging.info(f"Step6 (rerank): {time.time()-step:.2f}s")

        # 7. Initial answer
        step = time.time()
        ctx = "\n\n".join(d.page_content for d in top_docs)
        init_prompt = prompt.format(context=ctx, question=question)
        init_resp = llama_client.invoke(init_prompt)
        init_ans = getattr(init_resp, "content", str(init_resp))
        logging.info(f"Step7 (init answer): {time.time()-step:.2f}s")

        # 8. Step-Back
        step = time.time()
        sb_prompt = (f"Review this draft answer and suggest a follow-up retrieval query.\n\n"
                     f"Question: {question}\nAnswer: {init_ans}")
        sb_query = llama_client.invoke(sb_prompt).content
        more_docs = retriever.get_relevant_documents(sb_query)
        merged = dedupe(top_docs + more_docs)
        logging.info(f"Step8 (step-back): {time.time()-step:.2f}s")

        # 9. Final rerank & answer
        step = time.time()
        final_docs = rerank(merged, question, embed_fn, top_k=5)
        final_ctx = "\n\n".join(d.page_content for d in final_docs)
        final_prompt = prompt.format(context=final_ctx, question=question)
        final_ans = llama_client.invoke(final_prompt).content
        logging.info(f"Step9 (final answer): {time.time()-step:.2f}s")

        # Build response
        return ResponseModel(
            answer=final_ans.strip(),
            retrieved_documents=[
                DocumentInfo(page=str(d.metadata.get("page","N/A")),
                             link="Uploaded PDF",
                             snippet=d.page_content)
                for d in final_docs
            ]
        )

    except HTTPException:
        raise
    except Exception:
        logging.error("Exception in /ask:", exc_info=True)
        raise HTTPException(500, "Internal server error")
    finally:
        logging.info(f"Total time: {time.time() - total_start:.2f}s")
