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

from workflows.agent import create_ipl_agent

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(
    api_key=api_key
)

st.title("IPL Information Chatbot", anchor=False)
st.caption("Ask anything about the IPL. Answers come from the IPL document stored in the vector database.")

# Every question calls a paid API, so a public deployment is gated behind
# an access code (set ASSISTANT_ACCESS_CODE in .env or Streamlit secrets).
ASSISTANT_ACCESS_CODE = os.getenv("ASSISTANT_ACCESS_CODE")
if ASSISTANT_ACCESS_CODE and not st.session_state.get("unlocked", False):
    with st.form("unlock"):
        st.write("This chatbot calls a paid AI API. Enter the access code to use it.")
        code_input = st.text_input("Access code", type="password")
        if st.form_submit_button("Unlock"):
            if code_input == ASSISTANT_ACCESS_CODE:
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
    st.stop()

question = st.chat_input("Ask a question about the IPL")

file = st.file_uploader("Optional: upload another IPL PDF to add to the knowledge base")

if file:
    data = extract_text(file)
    st.success(f"Text extracted: {len(data)} characters")

    # Chunking: 1000-character chunks with 200-character overlap
    chunks = []
    start = 0
    while start < len(data):
        chunks.append(data[start:start + 1000])
        start += 800

    # =============== Method 1 Of Flooting Vector DB ===============
    # Langchain Document
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(page_content=chunk, metadata={
            "source": "ipl_wikipedia.pdf",
            "chunk_id": i
        }))

    # Langchain Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # pushing data 
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name="ipl_langchain", persist_directory="croma_langchain")
    st.success(f"Added {len(documents)} chunks to the knowledge base")

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

    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        agent = create_ipl_agent()
        res = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

    st.chat_message("assistant").write(res["messages"][-1].content)

    # # # =============== Searching Similarities (Langchan) =============== 
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vector_store = Chroma(embedding_function=embeddings, collection_name="ipl_langchain", persist_directory="croma_langchain")
    # retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    # docs = retriever.invoke(question)
    # # st.write("".join(doc.page_content for doc in docs))
    # context = "\n".join(doc.page_content for doc in docs)

    # # # =============== Searching Similarities (Manual) =============== 
    # # model = SentenceTransformer("all-MiniLM-L6-v2")
    # # dbclient = chromadb.PersistentClient(path="/croma_db")
    # # collection = dbclient.get_or_create_collection(name="ipl_info")
    # # query_vector = model.encode(question).tolist() 
    # # results = collection.query(
    # #     query_embeddings=[query_vector],
    # #     n_results=10
    # # )
    # # # st.write(results["documents"][0])
    # # context = "\n\n".join(results["documents"][0])
    
    # prompt = ipl_prompt(context, question)
    # # st.write(prompt)

    # response = client.responses.create(
    #     model="gpt-6-luna",
    #     input=prompt
    # )
    # st.write(response.output_text)

