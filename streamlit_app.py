import streamlit as st 
from openai import OpenAI
from dotenv import load_dotenv
from utils.pdf_utils import extract_text
from utils.prompt import ipl_prompt 
from sentence_transformers import SentenceTransformer
import os

# import sys
# import pysqlite3
# sys.module["sqlite3"] = pysqlite3

import chromadb 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(
    api_key=api_key
)

st.title("Hello", anchor=False) 

question = st.chat_input("question")

file = st.file_uploader("upload pdf")

if file:
    data = extract_text(file)
    st.success(f"data extracted successfully!! {len(data)}")

    # =============== Method 1 Of Flooting Vector DB =============== 
    # Langchain Document 
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(page_content=chunk, metadata={
            "source": "ipl_wikipedia.pdf",
            "chunk_id": i
        }))
    st.write(documents[0])

    # Langchain Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # pushing data 
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name="ipl_langchain", persist_directory="croma_langchain")
    st.success("pushed data to database")

    # # =============== Method 2 Of Flooting Vector DB =============== 
    # # Chunking
    # start = 0
    # chunks = []
    # while start < len(data):
    #     end = start + 1000
    #     chunk = data[start:end]
    #     chunks.append(chunk)
    #     start = end - 200
    # st.success(f"Chunking successful {len(chunks)}")
    # st.success(chunks)
    # # Embedding
    # embeded_chunks = []
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # for i, chunk in enumerate(chunks):
    #     vector = model.encode(chunk).tolist() 
    #     embeded_chunks.append({
    #         "id": i,
    #         "text": chunk,
    #         "embedding": vector
    #     })

    

    # st.success(f"Embedding successful {len(embeded_chunks)}")
    # # st.success(chunks)

    # # ------------------ pushing data to vector db ------------------ 

    # dbclient = chromadb.PersistentClient(path="/croma_db")
    # collection = dbclient.get_or_create_collection(name="ipl_info")

    # ids = []
    # docs = []
    # embeds = []
    # metadata = []
    # for item in embeded_chunks:
    #     ids.append(str(item["id"]))
    #     docs.append(item["text"])
    #     embeds.append(item["embedding"])
    #     metadata.append({
    #         "source": "ipl_wikipedia.pdf",
    #         "chunk_id": str(item["id"])
    #     })

    # collection.upsert(ids=ids, embeddings=embeds, metadatas=metadata, documents=docs)
    # st.success("DB Flooded")


if question:
    # # =============== Searching Similarities (Langchan) =============== 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(embedding_function=embeddings, collection_name="ipl_langchain", persist_directory="croma_langchain")
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    # st.write("".join(doc.page_content for doc in docs))
    context = "\n".join(doc.page_content for doc in docs)

    # # =============== Searching Similarities (Manual) =============== 
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # dbclient = chromadb.PersistentClient(path="/croma_db")
    # collection = dbclient.get_or_create_collection(name="ipl_info")
    # query_vector = model.encode(question).tolist() 
    # results = collection.query(
    #     query_embeddings=[query_vector],
    #     n_results=10
    # )
    # # st.write(results["documents"][0])
    # context = "\n\n".join(results["documents"][0])
    
    prompt = ipl_prompt(context, question)
    # st.write(prompt)

    response = client.responses.create(
        model="gpt-6-luna",
        input=prompt
    )
    st.write(response.output_text)

