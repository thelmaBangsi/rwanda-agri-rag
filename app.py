import sys
import os

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_PATH = "./data"
CHROMA_PATH = "./chroma_agri_db"

loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(CHROMA_PATH):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
else:
    vectorstore = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.0)

template = """You are an agricultural data assistant for Rwanda. 
Answer the question based only on the following retrieved context. 
If the information is not present in the context, state clearly that it is not available in official records.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

agri_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("Rwanda Agricultural RAG Pipeline")
    print("Type 'exit' or 'quit' to end session.")
    print("-" * 40)

    while True:
        try:
            user_query = input("\nQuery: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit', 'q']:
                break
            
            response = agri_rag_chain.invoke(user_query)
            print(f"\nAnswer: {response}")
        except (KeyboardInterrupt, EOFError):
            break
