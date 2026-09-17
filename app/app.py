"""
Streamlit demo app for the Research Paper Answer Bot.
Stretch Goal 2 (Advanced Option 2) — a polished, shareable UI over the RAG system
built and validated in Research_Paper_Answer_Bot.ipynb.

Run with:  streamlit run app.py
"""
import streamlit as st
from rag_core import RAGSystem

st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📄", layout="wide")

st.title("📄 Research Paper Answer Bot")
st.caption(
    "Ask questions about the indexed GenAI research papers. "
    "Answers are grounded in retrieved passages with paper title + page citations."
)


@st.cache_resource(show_spinner="Loading RAG system (embeddings, vector index, LLM)...")
def get_rag_system():
    return RAGSystem()


try:
    rag = get_rag_system()
    st.sidebar.success(f"LLM backend: {rag.llm_backend}")
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list of dicts for rendering

with st.sidebar:
    st.header("About")
    st.write(
        "This app answers questions using only the indexed research papers. "
        "If the answer isn't in the papers, it will say so rather than guessing."
    )
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

# Render prior turns
for turn in st.session_state.display_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("📚 Top supporting sources"):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**[{i}]** {s['paper_title']} — page {s['page']}")

query = st.chat_input("Ask a question about the research papers...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating answer..."):
            result = rag.answer(query, chat_history=st.session_state.chat_history)
        st.write(result["answer"])
        with st.expander("📚 Top supporting sources"):
            for i, s in enumerate(result["sources"], 1):
                st.markdown(f"**[{i}]** {s['paper_title']} — page {s['page']}")

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Assistant", result["answer"]))
    st.session_state.display_history.append(result)
