import os
import tempfile
import fitz
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=" AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN RESPONSIVE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(139, 92, 246, 0.08),
                transparent 25%
            ),
            #080d18;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #101827 0%,
                #0b1220 100%
            );

        border-right:
            1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 34px;

        margin-bottom: 24px;

        border-radius: 24px;

        border:
            1px solid rgba(255, 255, 255, 0.10);

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.92),
                rgba(15, 23, 42, 0.90)
            );

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .hero::before {
        content: "";

        position: absolute;

        width: 300px;
        height: 300px;

        border-radius: 50%;

        top: -180px;
        right: -100px;

        background:
            rgba(59, 130, 246, 0.18);

        filter: blur(20px);
    }

    .hero-badge {
        display: inline-block;

        padding: 7px 13px;

        margin-bottom: 15px;

        border-radius: 999px;

        background:
            rgba(59, 130, 246, 0.12);

        border:
            1px solid rgba(96, 165, 250, 0.25);

        color: #93c5fd;

        font-size: 13px;

        font-weight: 600;
    }

    .hero-title {
        position: relative;

        margin: 0;

        font-size: clamp(
            2rem,
            5vw,
            3.8rem
        );

        line-height: 1.05;

        font-weight: 800;

        letter-spacing: -2px;

        color: #f8fafc;
    }

    .hero-subtitle {
        position: relative;

        max-width: 800px;

        margin-top: 16px;

        margin-bottom: 0;

        color: #94a3b8;

        font-size: clamp(
            0.95rem,
            2vw,
            1.1rem
        );

        line-height: 1.7;
    }

    /* --------------------------------------------------------
       CARDS
    -------------------------------------------------------- */

    .glass-card {
        padding: 22px;

        border-radius: 18px;

        background:
            rgba(15, 23, 42, 0.72);

        border:
            1px solid rgba(255, 255, 255, 0.08);

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.18);

        margin-bottom: 16px;
    }

    .feature-card {
        min-height: 160px;

        padding: 22px;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.82),
                rgba(15, 23, 42, 0.82)
            );

        border:
            1px solid rgba(255, 255, 255, 0.08);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    .feature-card:hover {
        transform:
            translateY(-3px);

        border-color:
            rgba(96, 165, 250, 0.35);
    }

    .feature-icon {
        font-size: 28px;

        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 17px;

        font-weight: 700;

        color: #f8fafc;

        margin-bottom: 7px;
    }

    .feature-text {
        color: #94a3b8;

        font-size: 14px;

        line-height: 1.6;
    }

    /* --------------------------------------------------------
       STATUS
    -------------------------------------------------------- */

    .status-box {
        padding: 14px 16px;

        border-radius: 14px;

        background:
            rgba(34, 197, 94, 0.09);

        border:
            1px solid rgba(34, 197, 94, 0.25);

        color: #86efac;

        margin-bottom: 15px;
    }

    /* --------------------------------------------------------
       DOCUMENT CHIP
    -------------------------------------------------------- */

    .doc-chip {
        padding: 10px 12px;

        margin-bottom: 8px;

        border-radius: 11px;

        background:
            rgba(255, 255, 255, 0.04);

        border:
            1px solid rgba(255, 255, 255, 0.07);

        color: #cbd5e1;

        font-size: 13px;

        overflow-wrap: anywhere;
    }

    /* --------------------------------------------------------
       SOURCE CARD
    -------------------------------------------------------- */

    .source-card {
        padding: 14px;

        margin-bottom: 12px;

        border-radius: 13px;

        background:
            rgba(59, 130, 246, 0.06);

        border:
            1px solid rgba(59, 130, 246, 0.18);
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 12px;

        min-height: 45px;

        font-weight: 600;

        transition:
            transform 0.15s ease;
    }

    .stButton > button:hover {
        transform:
            translateY(-1px);
    }

    /* --------------------------------------------------------
       TABS
    -------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-weight: 600;

        padding-left: 16px;

        padding-right: 16px;
    }

    /* --------------------------------------------------------
       CHAT INPUT
    -------------------------------------------------------- */

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media screen and (max-width: 768px) {

        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 5rem;
        }

        .hero {
            padding: 23px 18px;
            border-radius: 18px;
        }

        .hero-title {
            letter-spacing: -1px;
        }

        .feature-card {
            min-height: auto;
        }

        .glass-card {
            padding: 16px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
    }

    /* --------------------------------------------------------
       SMALL MOBILE
    -------------------------------------------------------- */

    @media screen and (max-width: 480px) {

        .hero {
            padding: 20px 15px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }

        button[data-baseweb="tab"] {
            padding-left: 8px;
            padding-right: 8px;
            font-size: 12px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ENVIRONMENT + GEMINI
# ============================================================

load_dotenv(
    override=True
)

api_key = os.getenv(
    "GEMINI_API_KEY"
)


if not api_key:

    st.error(
        "❌ GEMINI_API_KEY was not found."
    )

    st.info(
        "Add GEMINI_API_KEY to your .env file "
        "and restart the application."
    )

    st.stop()


client = genai.Client(
    api_key=api_key
)


# This model worked in your notebook.
MODEL_NAME = "gemini-3-flash-preview"


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_embeddings():

    return HuggingFaceEmbeddings(

        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


embeddings = load_embeddings()


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS = {

    "vector_store": None,

    "messages": [],

    "document_names": [],

    "total_pages": 0,

    "total_chunks": 0,

    "page_documents": [],

    "all_chunks": [],

    "summary": "",

    "risk_analysis": "",

    "clause_analysis": ""
}


for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf(
    uploaded_file
):

    documents = []

    temp_path = None


    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name


        pdf_document = fitz.open(
            temp_path
        )


        total_pages = len(
            pdf_document
        )


        for page_number in range(
            total_pages
        ):

            page = pdf_document[
                page_number
            ]


            text = page.get_text(
                "text"
            )


            if text and text.strip():

                documents.append(

                    Document(

                        page_content=(
                            text.strip()
                        ),

                        metadata={

                            "source":
                                uploaded_file.name,

                            "page":
                                page_number + 1
                        }
                    )
                )


        pdf_document.close()


        return (
            documents,
            total_pages
        )


    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# PROCESS PDF DOCUMENTS
# ============================================================

def process_pdf_files(
    uploaded_files
):

    all_page_documents = []

    document_names = []

    total_pages = 0


    for uploaded_file in uploaded_files:

        documents, page_count = (
            extract_pdf(
                uploaded_file
            )
        )


        all_page_documents.extend(
            documents
        )


        document_names.append(
            uploaded_file.name
        )


        total_pages += (
            page_count
        )


    if not all_page_documents:

        raise ValueError(
            "No readable text was found "
            "in the uploaded PDF files."
        )


    splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    )


    chunks = (
        splitter.split_documents(
            all_page_documents
        )
    )


    vector_store = (
        FAISS.from_documents(

            documents=chunks,

            embedding=embeddings
        )
    )


    return {

        "vector_store":
            vector_store,

        "page_documents":
            all_page_documents,

        "chunks":
            chunks,

        "document_names":
            document_names,

        "total_pages":
            total_pages
    }


# ============================================================
# SOURCE LABEL
# ============================================================

def get_source_label(
    document
):

    source = (
        document.metadata.get(
            "source",
            "Unknown document"
        )
    )


    page = (
        document.metadata.get(
            "page",
            "Unknown"
        )
    )


    return (
        f"{source} — Page {page}"
    )


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(
    documents
):

    context_parts = []


    for index, document in enumerate(
        documents,
        start=1
    ):

        source = (
            document.metadata.get(
                "source",
                "Unknown"
            )
        )


        page = (
            document.metadata.get(
                "page",
                "Unknown"
            )
        )


        context_parts.append(

            f"""
==============================
SOURCE {index}
==============================

FILE: {source}
PAGE: {page}

{document.page_content}
"""
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def generate_with_gemini(
    prompt
):

    response = (
        client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt
        )
    )


    if not response.text:

        return (
            "The AI model did not return "
            "a text response."
        )


    return response.text


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    question
):

    if (
        st.session_state.vector_store
        is None
    ):

        raise ValueError(
            "Please process documents first."
        )


    retrieved_documents = (

        st.session_state
        .vector_store
        .similarity_search(

            question,

            k=min(
                6,
                len(
                    st.session_state
                    .all_chunks
                )
            )
        )
    )


    context = build_context(
        retrieved_documents
    )


    prompt = f"""
You are , an AI legal document assistant.

You must answer the user's question using ONLY
the retrieved document context below.

RULES:

1. Do not invent information.

2. If the answer cannot be found in the context,
say:

"I could not find enough information in the
uploaded documents to answer this question."

3. Explain the answer clearly.

4. Mention relevant document names and page numbers.

5. Distinguish between information from different
documents.

6. Do not claim to be a lawyer.

7. Do not provide a legal conclusion beyond what
the uploaded documents support.

8. Use readable Markdown.

9. End with a short note:

"Document-based AI analysis only — not professional
legal advice."


RETRIEVED DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    answer = generate_with_gemini(
        prompt
    )


    return (
        answer,
        retrieved_documents
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_summary():

    documents = (
        st.session_state
        .all_chunks
    )


    selected_documents = (
        documents[:20]
    )


    context = build_context(
        selected_documents
    )


    prompt = f"""
You are LexAI, an AI legal document assistant.

Create a professional structured summary using
ONLY the document context below.

Include the following sections when available:

# Executive Summary

## Document Overview

## Parties

## Purpose

## Important Dates

## Payment and Financial Terms

## Main Rights and Obligations

## Confidentiality

## Data Protection

## Intellectual Property

## Liability

## Termination

## Dispute Resolution

## Governing Law

## Important Observations

For important statements, mention the source
filename and page number.

Do not invent missing information.

If something is not found, clearly state that it
was not identified in the available context.

End with:

"Automated document summary — not professional
legal advice."


DOCUMENT CONTEXT:

{context}


SUMMARY:
"""


    return generate_with_gemini(
        prompt
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def generate_risk_analysis():

    query = """
    legal risks obligations liability indemnity
    termination penalties payment confidentiality
    data protection intellectual property dispute
    governing law unusual clauses unclear terms
    """


    documents = (

        st.session_state
        .vector_store
        .similarity_search(

            query,

            k=min(
                15,
                len(
                    st.session_state
                    .all_chunks
                )
            )
        )
    )


    context = build_context(
        documents
    )


    prompt = f"""
You are LexAI, an AI legal document review assistant.

Review ONLY the provided document context.

Identify potential issues using cautious,
non-conclusive language.

Use this structure:

# Risk Analysis

## Overall Review

## 🔴 Higher-Priority Review Points

## 🟠 Medium-Priority Review Points

## 🟢 Lower-Priority Observations

## Missing or Unclear Information

For every identified point include:

- Issue
- Why it may matter
- Document
- Page

Do not invent clauses.

Do not state that something is legally invalid
unless the document itself explicitly establishes it.

Use words such as:

- may
- could
- potentially
- consider reviewing

End with:

"Automated document review only — not professional
legal advice."


DOCUMENT CONTEXT:

{context}


ANALYSIS:
"""


    return generate_with_gemini(
        prompt
    )


# ============================================================
# CLAUSE ANALYSIS
# ============================================================

def generate_clause_analysis():

    query = """
    payment confidentiality termination liability
    intellectual property data protection dispute
    resolution governing law obligations warranties
    """


    documents = (

        st.session_state
        .vector_store
        .similarity_search(

            query,

            k=min(
                15,
                len(
                    st.session_state
                    .all_chunks
                )
            )
        )
    )


    context = build_context(
        documents
    )


    prompt = f"""
You are LexAI.

Extract and explain the important clauses from
the provided legal document context.

Use this format:

# Key Clause Analysis

For every clause:

## Clause Name

**What it says:**

Explain the clause in simple language.

**Why it matters:**

Explain its practical importance without giving
professional legal advice.

**Source:**

Document name and page number.

Focus on:

- Payment
- Confidentiality
- Data Protection
- Intellectual Property
- Liability
- Termination
- Dispute Resolution
- Governing Law
- Other important obligations

Only include clauses actually supported by the
document context.

Do not invent missing clauses.

End with:

"Automated clause explanation only — not
professional legal advice."


DOCUMENT CONTEXT:

{context}


CLAUSE ANALYSIS:
"""


    return generate_with_gemini(
        prompt
    )


# ============================================================
# RESET APPLICATION
# ============================================================

def reset_application():

    for key, value in (
        SESSION_DEFAULTS.items()
    ):

        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size: 1.65rem;
            font-weight: 800;
            margin-bottom: 5px;
        ">
            ⚖️ Ai legal Assistant
        </div>

        <div style="
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 20px;
        ">
            AI Legal Document Assistant
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_files = (
        st.file_uploader(

            "Upload Legal PDFs",

            type=[
                "pdf"
            ],

            accept_multiple_files=True,

            help=(
                "Upload one or more "
                "text-based PDF documents."
            )
        )
    )


    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} "
            f"document(s) selected"
        )


        if st.button(
            "⚡ Process Documents",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Reading and indexing "
                    "your documents..."
                ):

                    result = (
                        process_pdf_files(
                            uploaded_files
                        )
                    )


                    st.session_state.vector_store = (
                        result[
                            "vector_store"
                        ]
                    )


                    st.session_state.page_documents = (
                        result[
                            "page_documents"
                        ]
                    )


                    st.session_state.all_chunks = (
                        result[
                            "chunks"
                        ]
                    )


                    st.session_state.document_names = (
                        result[
                            "document_names"
                        ]
                    )


                    st.session_state.total_pages = (
                        result[
                            "total_pages"
                        ]
                    )


                    st.session_state.total_chunks = (
                        len(
                            result[
                                "chunks"
                            ]
                        )
                    )


                    st.session_state.messages = []

                    st.session_state.summary = ""

                    st.session_state.risk_analysis = ""

                    st.session_state.clause_analysis = ""


                st.success(
                    "Documents processed successfully."
                )


            except Exception as error:

                st.error(
                    f"Processing failed: {error}"
                )


    if (
        st.session_state.vector_store
        is not None
    ):

        st.divider()


        st.markdown(
            """
            ### Workspace
            """
        )


        metric_col_1, metric_col_2 = (
            st.columns(2)
        )


        metric_col_1.metric(

            "Files",

            len(
                st.session_state
                .document_names
            )
        )


        metric_col_2.metric(

            "Pages",

            st.session_state
            .total_pages
        )


        st.metric(

            "Knowledge Chunks",

            st.session_state
            .total_chunks
        )


        st.markdown(
            "#### Active Documents"
        )


        for document_name in (
            st.session_state
            .document_names
        ):

            st.markdown(

                f"""
                <div class="doc-chip">
                    📄 {document_name}
                </div>
                """,

                unsafe_allow_html=True
            )


    st.divider()


    if st.button(
        "🗑️ Reset Workspace",
        use_container_width=True
    ):

        reset_application()

        st.rerun()


    st.caption(
        "RAG • FAISS • HuggingFace • Gemini"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <section class="hero">
        <div class="hero-badge">AI-POWERED LEGAL DOCUMENT INTELLIGENCE</div>
        <h1 class="hero-title">Understand legal documents faster.</h1>
        <p class="hero-subtitle">
            Upload legal PDFs, ask document-grounded questions, generate summaries,
            review important clauses and identify areas that may require closer attention.
        </p>
    </section>
    """,
    unsafe_allow_html=True
)

st.warning(
    "⚠️ AI provides automated document-based information only. "
    "It is not a substitute for professional legal advice."
)


# ============================================================
# EMPTY STATE
# ============================================================

if st.session_state.vector_store is None:
    st.markdown("### What can AI do?")

    features = [
        ("💬", "Ask Questions", "Chat with your uploaded legal documents using natural language."),
        ("📝", "Smart Summary", "Generate structured summaries of important document information."),
        ("🔍", "Clause Review", "Extract and explain important contractual clauses."),
        ("⚠️", "Risk Review", "Highlight provisions that may deserve closer human review."),
    ]

    columns = st.columns(4)
    for column, (icon, title, text) in zip(columns, features):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "👈 Upload one or more PDF documents from the sidebar and click "
        "**Process Documents**."
    )
    st.stop()


# ============================================================
# READY STATUS
# ============================================================

st.markdown(
    f"""
    <div class="status-box">
        ✅ Knowledge base ready —
        <strong>{len(st.session_state.document_names)}</strong> document(s),
        <strong>{st.session_state.total_pages}</strong> page(s),
        <strong>{st.session_state.total_chunks}</strong> searchable chunks.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

(
    chat_tab,
    summary_tab,
    clauses_tab,
    risk_tab

) = st.tabs(

    [
        "💬 Chat",
        "📝 Summary",
        "📑 Clauses",
        "⚠️ Risk Review"
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "Chat with your documents"
    )


    st.caption(
        "Answers are generated from retrieved "
        "content in your uploaded PDFs."
    )


    if not st.session_state.messages:

        st.info(
            "Try asking: "
            "\"What are the termination conditions?\""
        )


    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message[
                "role"
            ]
        ):

            st.markdown(
                message[
                    "content"
                ]
            )


            if message.get(
                "sources"
            ):

                with st.expander(
                    "📚 View retrieved sources"
                ):

                    for source in (
                        message[
                            "sources"
                        ]
                    ):

                        st.markdown(

                            f"""
                            **📄 {source["file"]}**

                            **Page:** {source["page"]}
                            """
                        )


                        st.text(
                            source[
                                "preview"
                            ]
                        )


                        st.divider()


    question = st.chat_input(
        "Ask a question about your legal documents..."
    )


    if question:

        st.session_state.messages.append(

            {

                "role":
                    "user",

                "content":
                    question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        with st.chat_message(
            "assistant"
        ):

            try:

                with st.spinner(
                    "Searching your documents..."
                ):

                    answer, sources = (
                        answer_question(
                            question
                        )
                    )


                st.markdown(
                    answer
                )


                source_list = []

                seen_sources = set()


                for document in sources:

                    file_name = (
                        document.metadata.get(
                            "source",
                            "Unknown"
                        )
                    )


                    page = (
                        document.metadata.get(
                            "page",
                            "Unknown"
                        )
                    )


                    source_key = (

                        file_name,

                        page
                    )


                    if (
                        source_key
                        not in seen_sources
                    ):

                        source_list.append(

                            {

                                "file":
                                    file_name,

                                "page":
                                    page,

                                "preview":
                                    document
                                    .page_content[
                                        :700
                                    ]
                            }
                        )


                        seen_sources.add(
                            source_key
                        )


                with st.expander(
                    "📚 View retrieved sources"
                ):

                    for source in (
                        source_list
                    ):

                        st.markdown(

                            f"""
                            **📄 {source["file"]}**

                            **Page:** {source["page"]}
                            """
                        )


                        st.text(
                            source[
                                "preview"
                            ]
                        )


                        st.divider()


                st.session_state.messages.append(

                    {

                        "role":
                            "assistant",

                        "content":
                            answer,

                        "sources":
                            source_list
                    }
                )


            except Exception as error:

                st.error(
                    f"Unable to generate "
                    f"answer: {error}"
                )


# ============================================================
# SUMMARY TAB
# ============================================================

with summary_tab:

    st.subheader(
        "AI Document Summary"
    )


    st.write(
        "Generate a structured overview of "
        "the uploaded legal documents."
    )


    if st.button(
        "✨ Generate Summary",
        key="summary_button",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Creating document summary..."
            ):

                st.session_state.summary = (
                    generate_summary()
                )


        except Exception as error:

            st.error(
                f"Summary generation failed: "
                f"{error}"
            )


    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


        st.download_button(

            label="⬇️ Download Summary",

            data=(
                st.session_state
                .summary
            ),

            file_name=(
                "lexai_document_summary.txt"
            ),

            mime="text/plain",

            use_container_width=True
        )


# ============================================================
# CLAUSES TAB
# ============================================================

with clauses_tab:

    st.subheader(
        "Key Clause Analysis"
    )


    st.write(
        "Extract and explain important clauses "
        "from your uploaded documents."
    )


    if st.button(
        "📑 Analyze Key Clauses",
        key="clause_button",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Reviewing important clauses..."
            ):

                (
                    st.session_state
                    .clause_analysis
                ) = (
                    generate_clause_analysis()
                )


        except Exception as error:

            st.error(
                f"Clause analysis failed: "
                f"{error}"
            )


    if (
        st.session_state
        .clause_analysis
    ):

        st.markdown(
            st.session_state
            .clause_analysis
        )


        st.download_button(

            label=(
                "⬇️ Download Clause Analysis"
            ),

            data=(
                st.session_state
                .clause_analysis
            ),

            file_name=(
                "lexai_clause_analysis.txt"
            ),

            mime="text/plain",

            use_container_width=True
        )


# ============================================================
# RISK TAB
# ============================================================

with risk_tab:

    st.subheader(
        "AI Risk Review"
    )


    st.info(
        "The AI highlights document provisions "
        "that may deserve closer review. "
        "Risk labels are not legal conclusions."
    )


    if st.button(
        "🔍 Start Risk Review",
        key="risk_button",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing document provisions..."
            ):

                (
                    st.session_state
                    .risk_analysis
                ) = (
                    generate_risk_analysis()
                )


        except Exception as error:

            st.error(
                f"Risk analysis failed: "
                f"{error}"
            )


    if (
        st.session_state
        .risk_analysis
    ):

        st.markdown(
            st.session_state
            .risk_analysis
        )


        st.download_button(

            label=(
                "⬇️ Download Risk Analysis"
            ),

            data=(
                st.session_state
                .risk_analysis
            ),

            file_name=(
                "lexai_risk_analysis.txt"
            ),

            mime="text/plain",

            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 10px 0 20px 0;
    ">

        ⚖️ ai legal assistant

        <br>

        RAG + FAISS + HuggingFace Embeddings + Gemini

        <br><br>

        Automated document analysis only —
        not professional legal advice.

    </div>
    """,
    unsafe_allow_html=True
)