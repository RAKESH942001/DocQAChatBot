# DocQA Chatbot Model

An AI‑powered Document Q&A chatbot web application that lets users upload **any** PDF and interact in a WhatsApp‑style chat interface. Under the hood it uses a Retrieval‑Augmented Generation (RAG) pipeline with Ollama embeddings, ChromaDB for vector storage, and Groq/LLaMA for natural language responses.

---

## 🚀 Features

- **Dynamic PDF uploads**: Upload arbitrary PDFs at runtime without pre‑processing.
- **WhatsApp‑style chat UI**: Streamlit frontend with chat bubbles, timestamps, and a pinned file attachment.
- **Two‑stage RAG pipeline**:
  1. **Upload** endpoint to extract, chunk, embed and cache the PDF.
  2. **Ask** endpoint for fast similarity search and LLM‑powered answers.
- **Friendly error handling**: Bot‑style messages for invalid PDFs, empty content, network or server errors.
- **Session management**: Each upload generates a `session_id` so you can ask multiple questions without re‑uploading.

---

## System Architecture

![DocQA Chatbot Architecture](docqa_architecture.png)

1. **Upload** (`POST /upload`): User uploads a PDF, gets a `session_id`, and triggers background vector store creation.
2. **Extraction & Chunking**: `PyPDFLoader` reads pages; `RecursiveCharacterTextSplitter` breaks text into chunks.
3. **Embedding**: Ollama’s `nomic-embed-text` model converts chunks into vectors.
4. **Storage**: ChromaDB holds embeddings per `session_id` for fast lookup.
5. **Query** (`POST /ask`): Retrieves top‑k relevant chunks, builds a prompt, and calls Groq/LLaMA for the answer.
6. **Frontend**: Streamlit renders WhatsApp‑style chat bubbles, pinned “📄 File uploaded” message, timestamps, and bot‑style error messages.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Ollama installed and set up
- Groq API account and API key

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/RAKESH942001/DocQAChatBot.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download and set up Ollama:

   - Follow the instructions at [Ollama's official website](https://ollama.ai) to install Ollama
   - Download the Nomic text embedding model:
     ```
     ollama pull nomic-embed-text
     ```

4. Set up your Groq API key:
   - Create a `.env` file in the project root
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```
5. Start the FastAPI backend:

   ```
   uvicorn main:app --reload
   ```

6. Launch the Streamlit UI:

   ```
   streamlit run streamlit_app.py
   ```

7. Open your web browser and navigate to the Streamlit app URL (typically `http://localhost:8501`)

## Technical Summary

This project is a chatbot-style Document Q&A system where users upload PDFs and ask questions based on their content. The idea was to make it feel like chatting with a bot, similar to WhatsApp.

### Tech Used

- Frontend: Streamlit
- Backend: FastAPI
- PDF Parsing: pdfplumber
- Embeddings: nomic-embed-text + ChromaDB
- LLM: Groq (LLaMA 3-8B)
- Retrieval: LangChain

### Challenges Faced

- Some image-based PDFs had no readable text. I added checks to detect this and show a proper message.
- The chat UI initially re-rendered files multiple times. I used session handling to fix this.
- LangChain failed with empty embeddings, so I added a condition to prevent those errors.
- Faced GitHub permission issues but resolved it by updating my credentials.

### Improvements

With more time, I would:

- Add OCR for image-based PDFs.
- Improve the UI further using React or a custom frontend.
- Add user login and history.

This project helped me understand how to build a full AI app and manage real-world problems.
