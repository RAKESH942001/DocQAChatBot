from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import traceback

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever

from prompt import prompt
from llm import llm, compressor # your Groq LLaMA client & compressor


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

class ResponseModel(BaseModel):
    answer: str
    retrieved_documents: list[DocumentInfo]


@app.post("/ask", response_model=ResponseModel)
async def ask_question(
    question: str = Form(...),
    pdf: UploadFile = File(...)
):
    try:
        # 1. Save the uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            tmp_path = tmp.name

        # 2. Load PDF pages
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        # 3. Detect image-only (no selectable text)
        full_text = "".join([p.page_content for p in pages])
        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="⚠️ Could not extract text: this appears to be a scanned/image‑only PDF without selectable text."
            )

        # 4. Split into text chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        # filter out any empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="⚠️ No useful text chunks could be created from this PDF."
            )

        # 5. Build vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        try:
            vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="❌ Failed to generate embeddings. Ensure the PDF has readable text."
            )

        # 6. Create a compressed retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_db.as_retriever(search_kwargs={"k": 5})
        )
        retrieved_docs = retriever.get_relevant_documents(question)
        if not retrieved_docs:
            raise HTTPException(
                status_code=400,
                detail="⚠️ No relevant content found for your question."
            )

        # 7. Format document info
        doc_info = []
        for doc in retrieved_docs:
            page_num = doc.metadata.get("page", "N/A")
            doc_info.append(DocumentInfo(
                page=str(page_num),
                link="Uploaded PDF",
                snippet=doc.page_content
            ))

        # 8. Build prompt and invoke LLM
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        formatted_prompt = prompt.format(context=context, question=question)
        llm_resp = llm.invoke(formatted_prompt)
        answer = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)

        return ResponseModel(answer=answer, retrieved_documents=doc_info)

    except HTTPException:
        # Re‑raise our intentional 4xx errors
        raise
    except Exception as e:
        # Log and return a generic 500
        print("Exception in /ask:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="🚨 Internal server error while processing your request."
        )
