from langchain_groq import ChatGroq
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM client
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"  # You can change this to other Groq models
)

# Create a document compressor for better retrieval
compressor = LLMChainExtractor.from_llm(llm)