import logging
import os
import time
from typing import List, Tuple

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores.utils import cosine_similarity

from llm import llama_client

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def load_vector_db(vector_db_path: str) -> Chroma:
    """
    Load (or create) a persisted Chroma vector store.
    """
    logging.info(f"Loading vector DB from: {vector_db_path}")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    logging.info("Vector DB loaded.")
    return db

def build_multi_query_retriever(db: Chroma, k: int = 4, num_paraphrases: int = 3) -> MultiQueryRetriever:
    """
    Wrap the base similarity retriever with LangChain's MultiQueryRetriever.
    """
    logging.info("Building MultiQueryRetriever...")
    base_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llama_client,
        max_concurrent_queries=num_paraphrases,
        verbose=True
    )
    logging.info("MultiQueryRetriever ready.")
    return multi_retriever

def dedupe_documents(docs: List) -> List:
    """
    Remove duplicate page_content entries while preserving order.
    """
    logging.debug("Starting deduplication...")
    seen = set()
    unique = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            unique.append(doc)
            seen.add(text)
    logging.debug(f"Deduplicated from {len(docs)} to {len(unique)} documents.")
    return unique

def rerank_top_k(docs: List, query: str, embedding_fn, top_k: int = 8) -> List:
    """
    Light re‑ranking using cosine similarity to the original query.
    """
    logging.info("Starting reranking...")
    start = time.time()
    q_vec = embedding_fn.embed_query(query)
    scored = []
    for i, doc in enumerate(docs):
        d_vec = embedding_fn.embed_query(doc.page_content)
        score = cosine_similarity(q_vec, d_vec)
        scored.append((score, doc))
        logging.debug(f"Doc {i+1} score: {score:.4f}")
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:top_k]]
    logging.info(f"Reranking complete in {time.time() - start:.2f}s.")
    print(f"Reranking took {time.time() - start:.2f} seconds.")
    return top_docs

def query_llm(
    question: str,
    db: Chroma,
    llama_client,
    k: int = 5,
    num_paraphrases: int = 3
) -> Tuple[str, List]:
    """
    Full RAG pipeline with:
      • HyDE (hypothetical embedding)
      • Multi‑Query Retrieval + RAG Fusion
      • Reranking
      • Step‑Back Retrieval
    """
    try:
        total_start = time.time()
        print("Starting query_llm pipeline...")
        hyde_start = time.time()
        logging.info(f"[HyDE] Generating hypothetical answer for: {question}")
        hyde_prompt = f"Generate a concise, factual answer to help find relevant documents:\n\nQuestion: {question}\nAnswer:"
        hyde_resp = llama_client.invoke(hyde_prompt)
        hyde_answer = hyde_resp.content if hasattr(hyde_resp, "content") else str(hyde_resp)
        logging.debug(f"[HyDE] Answer draft: {hyde_answer}")
        print(f"HyDE generation took {time.time() - hyde_start:.2f} seconds.")
        retrieval_start = time.time()
        retriever = build_multi_query_retriever(db, k=k, num_paraphrases=num_paraphrases)
        logging.info("[Retriever] Running Multi‑Query Retrieval using HyDE answer...")
        docs_hyde = retriever.get_relevant_documents(hyde_answer)
        logging.info("[Retriever] Running Multi‑Query Retrieval on the original question for fusion...")
        docs_orig = retriever.get_relevant_documents(question)
        print(f"Retrieval took {time.time() - retrieval_start:.2f} seconds.")
        fusion_start = time.time()
        all_docs = docs_hyde + docs_orig
        all_docs = dedupe_documents(all_docs)
        logging.info(f"[Fusion] Total unique docs after fusion: {len(all_docs)}")
        print(f"Fusion and deduplication took {time.time() - fusion_start:.2f} seconds.")
        rerank_start = time.time()
        top_docs = rerank_top_k(all_docs, question, db._embedding_function, top_k=k)
        print(f"Reranking (top-k) took {time.time() - rerank_start:.2f} seconds.")
        initial_llm_start = time.time()
        context = "\n\n".join(d.page_content for d in top_docs)
        initial_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above:"
        )
        logging.info("[LLM] Generating initial answer...")
        initial_resp = llama_client.invoke(initial_prompt)
        initial_answer = initial_resp.content if hasattr(initial_resp, "content") else str(initial_resp)
        logging.debug(f"[Initial Answer] {initial_answer}")
        print(f"Initial LLM answer generation took {time.time() - initial_llm_start:.2f} seconds.")
        step_back_start = time.time()
        step_back_prompt = (
            f"You are reviewing the answer below and want to ensure completeness.\n\n"
            f"Question: {question}\n"
            f"Draft Answer: {initial_answer}\n\n"
            f"What follow‑up search query would you use to find any missing facts or verify this answer?"
        )
        logging.info("[Step‑Back] Generating follow‑up query...")
        sb_resp = llama_client.invoke(step_back_prompt)
        step_back_query = sb_resp.content.strip()
        logging.debug(f"[Step‑Back] Query: {step_back_query}")
        print(f"Step-back query generation took {time.time() - step_back_start:.2f} seconds.")
        step_back_retrieval_start = time.time()
        logging.info("[Step‑Back] Retrieving additional documents...")
        docs_sb = retriever.get_relevant_documents(step_back_query)
        all_docs = dedupe_documents(top_docs + docs_sb)
        logging.info(f"[Step‑Back] Docs after additional retrieval: {len(all_docs)}")
        print(f"Step-back retrieval and deduplication took {time.time() - step_back_retrieval_start:.2f} seconds.")
        final_rerank_start = time.time()
        final_docs = rerank_top_k(all_docs, question, db._embedding_function, top_k=k)
        print(f"Final reranking took {time.time() - final_rerank_start:.2f} seconds.")
        final_llm_start = time.time()
        final_context = "\n\n".join(d.page_content for d in final_docs)
        final_prompt = (
            f"Context:\n{final_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer accurately based on the context above:"
        )
        logging.info("[LLM] Generating final answer after step‑back retrieval...")
        final_resp = llama_client.invoke(final_prompt)
        final_answer = final_resp.content if hasattr(final_resp, "content") else str(final_resp)
        print(f"Final LLM answer generation took {time.time() - final_llm_start:.2f} seconds.")
        print(f"Total query_llm pipeline took {time.time() - total_start:.2f} seconds.")
        print(f"Final Answer: {final_answer.strip()}")
        return final_answer.strip(), final_docs
    except Exception as e:
        logging.error(f"query_llm error: {e}", exc_info=True)
        return "Sorry, something went wrong while processing your question.", []

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python query_engine.py <vector_db_path> <question>")
        sys.exit(1)
    db_path, user_q = sys.argv[1], sys.argv[2]
    db = load_vector_db(db_path)
    answer, docs = query_llm(user_q, db, llama_client)
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("\n=== DOCUMENTS ===")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content[:200]}...")
