from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# If you use Groq API, import requests or use groq's SDK
# import requests

def load_vector_db(vector_db_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    return db

def query_llm(db, question: str):
    docs = db.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the following text:
------------------
{context}
------------------
Question: {question}
If answer not found, say "Not in document.""""

    # Step 4: Send to Groq or Ollama or OpenAI
    result = fake_llm(prompt)  # ← Replace with actual call
    return result, docs
