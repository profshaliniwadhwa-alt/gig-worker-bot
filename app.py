"""
Gig Worker Law Assistant (Streamlit).
Answers questions about the gig-worker laws of Rajasthan, Karnataka, Telangana and Jharkhand in
English, Hindi, Hinglish or Tamil, using retrieval-augmented generation:
  PDFs in policies/ -> 800-character passages labelled with their source law ->
  multilingual MiniLM embeddings -> FAISS -> top-3 passages -> gpt-oss-20b on Groq.
The Groq API key is read from Streamlit secrets (GROQ_API_KEY) or the environment / .env file.
"""
import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

PDF_DIR = "policies"                      # every PDF here is indexed; file name = source label
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-20b"          # llama-3.1-8b-instant was retired by Groq on 16 Aug 2026
REASONING_EFFORT = "low"
DEMO_DRY = os.environ.get("DEMO_DRY") == "1"   # offline testing only

PROMPT_TEMPLATE = (
    "You are a professional assistant for gig and platform workers in India, "
    "specializing in the gig-worker welfare laws of Indian states.\n"
    "Users may ask in any language. Respond in the same language while maintaining "
    "accuracy, completeness, and professionalism.\n\n"
    "If the answer cannot be derived from the context, respond with: 'Sorry, I don’t know.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n\n"
    "Answer:"
)

EXAMPLES = [
    "How much notice must a platform give before terminating a gig worker in Telangana?",
    "How much welfare fee do platforms have to pay in Karnataka?",
    "राजस्थान में एग्रीगेटर कल्याण शुल्क देर से जमा करे तो कितना ब्याज लगता है?",
    "Kya Karnataka mein platform ko gig workers ke liye ek human contact person dena hota hai?",
    "கர்நாடகாவில் தளங்கள் எவ்வளவு நலக் கட்டணம் செலுத்த வேண்டும்?",
    "What does the Maharashtra gig workers law say about registration?",
]


def get_api_key():
    load_dotenv()
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def law_name(path):
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ")


@st.cache_resource(show_spinner=False)
def build_pipeline():
    """Load the laws, label and embed the passages, and set up retrieval + generation (runs once)."""
    files = sorted(os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    docs = []
    for f in files:
        docs.extend(PyPDFLoader(f).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150,
                                              separators=["\n\n", "\n", ".", "?", "!", " "])
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["law"] = law_name(c.metadata.get("source", ""))
        c.page_content = f"[Source: {c.metadata['law']}]\n{c.page_content}"

    if DEMO_DRY:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        embed = DeterministicFakeEmbedding(size=384)
        llm = FakeListChatModel(responses=["(test answer)"] * 1000)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        llm = ChatGroq(model=LLM_MODEL, groq_api_key=get_api_key(), reasoning_effort=REASONING_EFFORT)

    retriever = FAISS.from_documents(chunks, embed).as_retriever(search_kwargs={"k": 3})
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm
    return retriever, chain, len(files), len(chunks)


def answer_question(retriever, chain, question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    text = chain.invoke({"context": context, "input": question}).content
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    sources = []
    for d in docs:
        body = re.sub(r"^\[Source:[^\]]*\]\n", "", d.page_content)
        body = re.sub(r"\s+", " ", body).strip()
        sources.append({"law": d.metadata.get("law", ""), "page": (d.metadata.get("page") or 0) + 1, "text": body})
    return text, sources


# ============================== UI ==============================
st.set_page_config(page_title="Gig Worker Law Assistant", page_icon="⚖️", layout="centered")
st.title("Gig Worker Law Assistant")
st.caption("Ask about the gig-worker laws of **Rajasthan, Karnataka, Telangana or Jharkhand** in English, "
           "Hindi, Hinglish or Tamil. Name the state in your question. Information only, not legal advice.")

if not DEMO_DRY and not get_api_key():
    st.error("No Groq API key found. In Streamlit Cloud, add GROQ_API_KEY under App settings → Secrets.")
    st.stop()

with st.spinner("Loading the four state laws and building the search index (first start takes about a minute)..."):
    retriever, chain, n_laws, n_chunks = build_pipeline()

with st.sidebar:
    st.subheader("About")
    st.write(f"Knowledge base: {n_laws} state laws, {n_chunks} passages.")
    st.write("Each answer is generated only from the 3 passages shown under it.")
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("Clear chat"):
        st.session_state["history"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        with st.expander("Law passages this answer is based on"):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**{i}. {s['law']}, page {s['page']}**")
                st.caption(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))

question = st.chat_input("Type your question here...") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the laws..."):
            try:
                ans, srcs = answer_question(retriever, chain, question)
            except Exception as e:
                ans, srcs = f"Sorry, something went wrong: {str(e)[:300]}", []
        st.write(ans)
        with st.expander("Law passages this answer is based on", expanded=True):
            for i, s in enumerate(srcs, 1):
                st.markdown(f"**{i}. {s['law']}, page {s['page']}**")
                st.caption(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))
    st.session_state["history"].append({"q": question, "a": ans, "sources": srcs})
