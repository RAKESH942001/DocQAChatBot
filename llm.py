from langchain_groq import ChatGroq
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
from dotenv import load_dotenv

load_dotenv()

llama_client = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

compressor = LLMChainExtractor.from_llm(llama_client)
