import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Load environment variables from.env file
load_dotenv()

# ---- 1. Load PDF ----
pdf_path = "data/DPMP-461.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()  # List of Document objects

# # ---- 2. Chunk Text ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                          chunk_overlap=100,
                                          separators=["\n\n", "\n", ".", " ", ""])
chunks = splitter.split_documents(documents)  # List of Document objects

# ---- 3. Set Up SambaNova Embeddings ----
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ---- 4. Build FAISS Vector Store ----
db = FAISS.from_documents(chunks, embeddings)
# # Save the vector store for later use
# db.save("vector_store.faiss")
# # Load the vector store if needed
# db = FAISS.load_local("vector_store.faiss", embeddings, allow_dangerous_deserialization=True)

# ---- 5. Retrieval Example ----
query = "Methods for x-ray imaging and image processing"
results = db.similarity_search(query, k=8)
print("\n--- Top Relevant Chunks ---")
for i, doc in enumerate(results, 1):
    print(f"\nChunk {i}:\n{doc.page_content}")

# # ---- 6. Use results in your LLM prompt ----
# context = "\n".join(doc.page_content for doc in results)
# # Now you can feed 'context' to your enhancer or LLM prompt