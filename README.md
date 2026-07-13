# ⚖️ AI Legal Document Assistant

An AI-powered legal document question-answering application built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload a PDF legal document, processes the document into searchable text chunks, retrieves the most relevant sections using semantic search, and uses Google Gemini to generate document-grounded answers.

> ⚠️ **Disclaimer:** This application provides document-based information for educational and demonstration purposes only. It is not a substitute for professional legal advice.

---

## 🚀 Features

- 📄 Upload PDF legal documents
- 🔍 Extract text from PDF files
- ✂️ Split documents into semantic chunks
- 🧠 Generate text embeddings
- 🗄️ Store and search embeddings using FAISS
- 🔎 Retrieve relevant document sections
- 🤖 Generate answers using Google Gemini
- 📚 Display document source and page references
- 💬 Interactive question-and-answer interface
- 🖥️ Modern Streamlit user interface
- 📱 Responsive design for desktop and mobile devices

---

## 🧠 How It Works

The application uses a **RAG (Retrieval-Augmented Generation)** pipeline:

```text
User Uploads PDF
        │
        ▼
Extract Text with PyMuPDF
        │
        ▼
Split Text into Chunks
        │
        ▼
Generate HuggingFace Embeddings
        │
        ▼
Store Vectors in FAISS
        │
        ▼
User Asks a Question
        │
        ▼
Semantic Similarity Search
        │
        ▼
Retrieve Relevant Document Chunks
        │
        ▼
Send Context + Question to Gemini
        │
        ▼
Generate Document-Grounded Answer
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| LangChain | Document processing |
| HuggingFace | Text embeddings |
| Sentence Transformers | Embedding model |
| FAISS | Vector similarity search |
| Google Gemini | Large Language Model |
| python-dotenv | Environment variable management |

---

## 📁 Project Structure

```text
AI-Legal-Assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
└── data/
    └── sample_contract.pdf
```

### File Description

- `app.py` — Main Streamlit application
- `README.md` — Project documentation
- `requirements.txt` — Required Python packages
- `.env` — Stores the Gemini API key locally
- `.gitignore` — Prevents sensitive and unnecessary files from being uploaded
- `data/` — Optional directory for sample legal documents

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Legal-Assistant.git
cd AI-Legal-Assistant
```

---

### 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n legal_ai python=3.11
conda activate legal_ai
```

Or using Python:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Gemini API Key Setup

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

The project should look like:

```text
AI-Legal-Assistant/
├── app.py
├── .env
└── ...
```

> 🔒 Never upload your `.env` file or API key to GitHub.

---

## ▶️ Run the Application

Activate your environment:

```bash
conda activate legal_ai
```

Go to the project folder:

```bash
cd /d D:\Project\AI-Legal-Assistant
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 📖 How to Use

1. Start the Streamlit application.
2. Upload a legal PDF document.
3. Click **Process Document**.
4. Wait for the PDF to be processed.
5. Enter a question about the uploaded document.
6. The system retrieves relevant sections.
7. Gemini generates an answer based on the retrieved context.
8. Review the source and page references.

### Example Questions

```text
What is the termination clause?
```

```text
How much must the client pay?
```

```text
What is the duration of the agreement?
```

```text
What does the confidentiality clause say?
```

```text
Which law governs this agreement?
```

---

## 🔄 RAG Architecture

```text
                    ┌─────────────────────┐
                    │    Legal PDF        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  PDF Text Extraction│
                    │      PyMuPDF        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Chunking     │
                    │     LangChain       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Embeddings      │
                    │    HuggingFace      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Vector Database   │
                    │       FAISS         │
                    └──────────┬──────────┘
                               │
             User Question ────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Context + Question  │
                    │       Gemini        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Grounded AI Answer  │
                    │ + Source References │
                    └─────────────────────┘
```

---

## 📦 requirements.txt

Example dependencies:

```text
streamlit
pymupdf
python-dotenv
google-genai
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
sentence-transformers
faiss-cpu
```

Generate the exact dependencies from your environment with:

```bash
pip freeze > requirements.txt
```

---

## 🔒 .gitignore

Create a `.gitignore` file:

```gitignore
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# FAISS indexes
*.faiss
*.pkl
```

---

## 🔐 Security

- Never commit your Gemini API key.
- Keep API credentials inside `.env`.
- Do not upload confidential legal documents to a public repository.
- Validate uploaded files before processing them.
- Consider file-size limits for production deployment.
- Use secure secret management when deploying the application.

---

## 🧪 Example Workflow

```text
Upload: sample_contract.pdf

Question:
How can the agreement be terminated?

Retrieved Context:
Relevant termination clauses from the document.

AI Answer:
The agreement may be terminated by providing the required written
notice or immediately in cases of material breach, according to the
terms contained in the uploaded document.

Source:
sample_contract.pdf — Page 2
```

---

## 🌟 Future Improvements

- [ ] Multiple PDF document support
- [ ] Conversation history
- [ ] PDF preview
- [ ] OCR support for scanned legal documents
- [ ] Multiple language support
- [ ] Tamil legal document support
- [ ] Legal document summarization
- [ ] Clause extraction
- [ ] Contract risk detection
- [ ] Document comparison
- [ ] User authentication
- [ ] Persistent vector database
- [ ] Export AI answers as PDF
- [ ] Cloud deployment
- [ ] Improved citation verification

---

## ⚠️ Disclaimer

This project is intended for:

- Educational purposes
- AI/ML learning
- RAG experimentation
- Portfolio demonstration
- Document analysis research

The generated responses may contain errors or omissions.

**Always consult a qualified legal professional for actual legal advice.**

---

## 👨‍💻 Author

**Naveen Baskar**

AI / Machine Learning Project

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
