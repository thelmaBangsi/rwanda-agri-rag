import sys, types
sys.modules["posthog"] = types.ModuleType("posthog")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load documents from ./data/
print("Loading documents from ./data...")
loader = DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(raw_docs)
print(f"Total document chunks generated: {len(docs)}")

# 3. Embed and store in ChromaDB
print("Generating HuggingFace embeddings and populating ChromaDB...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory="./chroma_agri_db"
)

# 4. Configure Retriever (k=2) & Chain (temperature=0.0)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

prompt_template = """Answer strictly based ONLY on the provided context. If the information is missing, respond: "I cannot find this information in official Rwandan agricultural records."

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.0)

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

agri_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute test suite
if __name__ == "__main__":
    queries = [
        "What percentage of farmers applied chemical fertilizer in Season A 2025?",
        "What was the total national milk production recorded by MINAGRI?",
        "How much revenue was generated from tea exports in 2023/2024?",
        "What subsidy does the government provide on small-scale irrigation equipment?",
        "When does the coffee harvesting period begin in Rwanda?"
    ]
    
    print("\n=== RUNNING PROTOTYPE TESTS ===")
    for q in queries:
        print(f"\nQ: {q}")
        response = agri_rag_chain.invoke(q)
        print(f"A: {response}")
