# Required imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Your Google API Key (you can load this from an env variable in production)
GOOGLE_API_KEY = "Your_Google_API_Key"

# Step 1: Load and split the document
loader = PyPDFLoader(r"D:\Desktop\Data Science\Excel Project 1 Report PDF.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 2: Create Embeddings with Google Generative AI
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Step 3: Build vector store with FAISS
vectorstore = FAISS.from_documents(chunks, embedding)

# Step 4: Configure Gemini model
genai.configure(api_key="Your_Google_API_Key")
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")


# Step 5: Define function to answer questions using Gemini + retrieved context
def answer_with_gemini(query: str) -> str:
    # Search relevant chunks
    relevant_docs = vectorstore.similarity_search(query, k=4)

    # Construct context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate response using Gemini
    response = gemini_model.generate_content(prompt)

    return response.text if hasattr(response, "text") else str(response)


# Run a sample query
query = "What are the key clauses in this document?"
print(answer_with_gemini(query))

# genai.configure(api_key="AIzaSyDwY1Xeq9MXc_qJUyfQ0-cfjQVvZbzUTuY")
# models = genai.list_models()
# for m in models:
#     print(m.name, m.supported_generation_methods)