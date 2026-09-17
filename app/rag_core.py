"""
rag_core.py
Shared RAG logic used by the Streamlit app.
"""
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

def get_google_api_key():
    """Retrieve Google API key from environment or Streamlit secrets."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = key
            return key
    except Exception:
        pass
    return None


def _fix_chroma_legacy_metadata(persist_dir):
    """
    Ensures legacy ChromaDB collections (created with earlier versions of ChromaDB)
    can be loaded by ChromaDB 0.5.20+ without KeyError: '_type' or dimension errors.
    Leaves all vectors, chunks, and documents untouched.
    """
    if not os.path.exists(persist_dir):
        return

    # 1. Ensure config_json_str in collections table has valid format
    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(db_file):
        try:
            import sqlite3
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name, config_json_str FROM collections")
            rows = cursor.fetchall()
            for name, cfg in rows:
                if not cfg or cfg == "{}" or "_type" not in cfg:
                    default_cfg = (
                        '{"hnsw_configuration": {"space": "l2", "ef_construction": 100, '
                        '"ef_search": 10, "num_threads": 8, "M": 16, "resize_factor": 1.2, '
                        '"batch_size": 100, "sync_threshold": 1000, '
                        '"_type": "HNSWConfigurationInternal"}, '
                        '"_type": "CollectionConfigurationInternal"}'
                    )
                    cursor.execute(
                        "UPDATE collections SET config_json_str = ? WHERE name = ?",
                        (default_cfg, name),
                    )
            conn.commit()
            conn.close()
        except Exception:
            pass

    # 2. Ensure index_metadata.pickle is wrapped in PersistentData with dimensionality
    for root, _, files in os.walk(persist_dir):
        if "index_metadata.pickle" in files:
            p_file = os.path.join(root, "index_metadata.pickle")
            try:
                import pickle
                with open(p_file, "rb") as f:
                    d = pickle.load(f)
                if isinstance(d, dict) or getattr(d, "dimensionality", None) is None:
                    from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData
                    dim = 384
                    if isinstance(d, dict):
                        p_data = PersistentData(
                            dimensionality=dim,
                            total_elements_added=d.get("total_elements_added", 0),
                            id_to_label=d.get("id_to_label", {}),
                            label_to_id=d.get("label_to_id", {}),
                            id_to_seq_id=d.get("id_to_seq_id", {}),
                        )
                    else:
                        d.dimensionality = dim
                        p_data = d
                    with open(p_file, "wb") as f:
                        pickle.dump(p_data, f)
            except Exception:
                pass


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(_BASE_DIR, "notebook", "data", "papers"))
PERSIST_DIR = os.environ.get("RAG_PERSIST_DIR", os.path.join(_BASE_DIR, "notebook", "chroma_db"))

RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions about a set of AI/ML research papers.

Answer the question using ONLY the context passages below. Each passage is labeled with a source number.
If the context does not contain enough information to answer, say "I don't know based on the provided context."
Do not use outside knowledge. When you make a claim, refer to it by its source number, e.g. (Source 1).

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"[Source {i}] ({d.metadata['paper_title']}, page {d.metadata['page']})\n{d.page_content}"
        )
    return "\n\n".join(blocks)


def extract_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content).strip()


class RAGSystem:
    def __init__(self, api_key=None):
        self.api_key = api_key or get_google_api_key()
        self.embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.vectorstore = None
        self.llm = None
        self.llm_backend = None
        self._load_or_build_index()
        self._load_llm()

    def _load_or_build_index(self):
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            _fix_chroma_legacy_metadata(PERSIST_DIR)
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embedder,
                collection_name="research_papers",
            )
            return

        pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDFs found in {DATA_DIR}. Run the notebook's data download cell first, "
                "or drop PDFs into that folder."
            )

        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load()
            paper_title = os.path.basename(path).replace("_", " ").replace(".pdf", "").title()
            for p in pages:
                p.metadata["paper_title"] = paper_title
                p.metadata["page"] = p.metadata.get("page", 0) + 1
            docs.extend(pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedder,
            persist_directory=PERSIST_DIR,
            collection_name="research_papers",
        )

    def _load_llm(self):
        api_key = self.api_key or get_google_api_key()
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please configure GOOGLE_API_KEY in your "
                "Streamlit Cloud Secrets or environment variables."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=api_key,
        )
        self.llm_backend = "gemini"

    def _rerank_retrieve(self, query, k=3, fetch_k=10):
        candidates = self.vectorstore.similarity_search(query, k=fetch_k)
        pairs = [[query, c.page_content] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def condense_question(self, question, history):
        if not history:
            return question
        history_str = "\n".join([f"{role}: {text}" for role, text in history[-6:]])
        condensed_prompt = (
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up question to be a standalone question that includes all necessary "
            f"context.\n\nChat History:\n{history_str}\n\nFollow-up Question: {question}\n\n"
            "Standalone Question:"
        )
        result = self.llm.invoke(condensed_prompt)
        return extract_text(result.content)

    def answer(self, question, chat_history=None, k=3):
        standalone_q = self.condense_question(question, chat_history or [])
        retrieved_docs = self._rerank_retrieve(standalone_q, k=k)
        context_str = format_context(retrieved_docs)
        chain_input = self.prompt.format(context=context_str, question=standalone_q)

        result = self.llm.invoke(chain_input)
        response_text = extract_text(result.content)

        return {
            "question": question,
            "standalone_question": standalone_q,
            "answer": response_text,
            "sources": [
                {"paper_title": d.metadata["paper_title"], "page": d.metadata["page"]}
                for d in retrieved_docs
            ],
        }