import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from utils.pdf_utils import extract_text
from workflows.agent import create_ipl_agent
from workflows.tools import vector_store, SOURCE_SEPARATOR

st.set_page_config(page_title="IPL Information Chatbot", page_icon="🏏")

st.title("IPL Information Chatbot", anchor=False)
st.caption(
    "A RAG chatbot: your question is matched against chunks of an IPL document "
    "stored in a vector database, and the AI answers only from those chunks."
)

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


# ---------------- Knowledge base (already loaded) ----------------

@st.cache_data(ttl=60)
def knowledge_base_summary():
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]
    counts = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


sources = knowledge_base_summary()
with st.container(border=True):
    st.markdown("**📚 Knowledge base (already loaded)**")
    for name, count in sources.items():
        st.write(f"📄 `{name}`: {count} chunks")
    st.caption("Answers come only from these documents. No upload needed.")

with st.expander("Add another IPL PDF (optional)"):
    file = st.file_uploader("Upload a PDF", type="pdf")
    if file and st.button("Add to knowledge base"):
        data = extract_text(file)

        # Chunking: 1000-character chunks with 200-character overlap
        chunks = []
        start = 0
        while start < len(data):
            chunks.append(data[start:start + 1000])
            start += 800

        documents = [
            Document(page_content=chunk, metadata={"source": file.name, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        vector_store.add_documents(documents)
        knowledge_base_summary.clear()
        st.success(f"Added {len(documents)} chunks from {file.name}")
        st.rerun()


# ---------------- Helpers ----------------

def message_text(message):
    """Turn an AI message's content into plain text (it can be a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def retrieved_sources(messages):
    """Collect the chunks the search tool returned while answering."""
    found = []
    for message in messages:
        if getattr(message, "type", "") == "tool" and message.name == "search_ipl_knowledge":
            for block in str(message.content).split(SOURCE_SEPARATOR):
                header, _, text = block.partition("\n")
                found.append((header.strip("[]"), text.strip()))
    return found


def show_sources(found):
    if not found:
        st.caption("⚠️ No document chunks were used for this answer.")
        return
    with st.expander(f"📎 Sources: {len(found)} chunks used"):
        for header, text in found:
            st.markdown(f"**{header}**")
            st.text(text[:600] + ("..." if len(text) > 600 else ""))


@st.cache_resource
def get_agent():
    return create_ipl_agent()


# ---------------- Chat ----------------

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        if item["role"] == "assistant":
            show_sources(item["sources"])

SAMPLE_QUESTIONS = [
    "Which team has won the most IPL titles?",
    "When was the first IPL season played?",
    "Who won the IPL in 2024?",
]

clicked = None
if not st.session_state.history:
    st.write("Try a sample question:")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, sample in zip(cols, SAMPLE_QUESTIONS):
        if col.button(sample, width="stretch"):
            clicked = sample

question = st.chat_input("Ask a question about the IPL") or clicked

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the IPL document..."):
            result = get_agent().invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
        answer = message_text(result["messages"][-1])
        found = retrieved_sources(result["messages"])
        st.markdown(answer)
        show_sources(found)

    st.session_state.history.append({"role": "user", "text": question})
    st.session_state.history.append({"role": "assistant", "text": answer, "sources": found})
    st.rerun()
