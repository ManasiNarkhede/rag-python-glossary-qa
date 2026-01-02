# 🐍 RAG Python Glossary Q&A Bot

A lightweight **Retrieval-Augmented Generation (RAG)** assistant that answers questions about Python terminology using the official **Python 3.12 Glossary** as its knowledge base.

---

## 📖 Overview

This project demonstrates a complete RAG pipeline that:

1. **Extracts** text from the Python Glossary PDF
2. **Chunks** the text into semantic segments using spaCy NLP
3. **Embeds** chunks using HuggingFace's `all-MiniLM-L6-v2` model
4. **Stores** embeddings in a persistent ChromaDB vector database
5. **Retrieves** relevant context based on user queries
6. **Generates** accurate answers using Llama 3.1 8B via OpenRouter API

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Python 3.12    │────▶│   PyPDF2     │────▶│  Raw Text       │
│  Glossary PDF   │     │  Extraction  │     │                 │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Sentence       │◀────│   spaCy      │◀────│  Text Chunking  │
│  Chunks         │     │  NLP Engine  │     │  (350 chars)    │
└────────┬────────┘     └──────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Vector         │────▶│  ChromaDB    │────▶│  Persistent     │
│  Embeddings     │     │  Storage     │     │  glossary_db/   │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │
         │  Query Time
         ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  User Question  │────▶│  Semantic    │────▶│  Top-5 Relevant │
│                 │     │  Search      │     │  Chunks         │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Final Answer   │◀────│  Llama 3.1   │◀────│  Context +      │
│                 │     │  8B (LLM)    │     │  Question       │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

---

## ✨ Features

- 🔍 **Semantic Search** – Finds the most relevant glossary entries for your question
- 🧠 **LLM-Powered Answers** – Uses Llama 3.1 8B Instruct for accurate, contextual responses
- 💾 **Persistent Storage** – ChromaDB saves embeddings locally (no re-processing on restart)
- ⚡ **Fast & Lightweight** – Runs locally with minimal resources
- 🔒 **Privacy-Focused** – Your data stays local; only queries go to the LLM API
- 📚 **Authoritative Source** – Answers directly from official Python documentation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **PDF Extraction** | PyPDF2 |
| **Text Chunking** | spaCy (en_core_web_sm) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **LLM Provider** | OpenRouter API |
| **LLM Model** | Meta Llama 3.1 8B Instruct |
| **Environment** | Python 3.8+, Jupyter Notebook |

---

## 📋 Prerequisites

- **Python 3.9+** installed
- **OpenRouter API Key** ([Get one free here](https://openrouter.ai/))
- **Python 3.12 Glossary PDF** (download from [Python Docs](https://docs.python.org/3/glossary.html))

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-python-glossary-qa.git
cd rag-python-glossary-qa
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 6. Add the Python Glossary PDF

Place the `python_glossary.pdf` file in the project root directory.

> 💡 **Tip:** You can download/export the glossary from [Python's Official Documentation](https://docs.python.org/3/glossary.html)

### 7. Run the Notebook

Open `RAG_QA_system.ipynb` in Jupyter Notebook or VS Code and run all cells.

```bash
jupyter notebook RAG_QA_system.ipynb
```

---

## 💬 Usage

Once the notebook is running, you'll see an interactive prompt:

```
======================================================================
     Rag-Python Glossary Q&A Bot is READY!     
     Powered by Llama 3.1 8B Instruct       
     Type 'quit', 'exit', or 'bye' to stop  
======================================================================

You: What is a decorator in Python?
RAG-Bot: A decorator is a function returning another function, usually applied 
as a function transformation using the @wrapper syntax...

You: Explain the GIL
RAG-Bot: The Global Interpreter Lock (GIL) is a mutex that protects access to 
Python objects, preventing multiple threads from executing Python bytecodes 
at once...

You: quit
Happy learning! Come back anytime.
```

---

## 📁 Project Structure

```
rag-python-glossary-qa/
│
├── RAG_QA_system.ipynb    # Main Jupyter notebook with RAG pipeline
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── .env                   # API keys (create this file - not tracked)
├── python_glossary.pdf    # Source PDF (add this file)
│
└── glossary_db/           # ChromaDB persistent storage (auto-generated)
    └── ...                # Vector embeddings & metadata
```

---

## 📦 Dependencies

```txt
openai>=1.30.0,<3.0.0      # OpenAI-compatible client for OpenRouter
python-dotenv>=1.0.0       # Environment variable management
faiss-cpu>=1.7.4           # Vector similarity search (optional backend)
chromadb>=0.4.24           # Vector database for embeddings
sentence-transformers==5.1.2  # HuggingFace embedding models
spacy==3.7.4               # NLP for text chunking
PyPDF2>=3.0.0              # PDF text extraction
```

---

## ⚙️ Configuration Options

You can customize these parameters in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CHUNK_LEN` | 350 | Maximum characters per text chunk |
| `n_results` | 5 | Number of relevant chunks to retrieve |
| `temperature` | 0.1 | LLM creativity (lower = more precise) |
| `max_tokens` | 500 | Maximum response length |
| `model` | llama-3.1-8b-instruct | LLM model via OpenRouter |

---

## 🔧 Troubleshooting

### ❌ "API key not found"
- Ensure `.env` file exists with `OPENROUTER_API_KEY=your_key`
- Verify the key is valid at [OpenRouter Dashboard](https://openrouter.ai/)

### ❌ "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### ❌ "PDF file not found"
- Place `python_glossary.pdf` in the project root directory

### ❌ Slow first run
- Initial embedding of chunks takes time (1-2 minutes)
- Subsequent runs load from `glossary_db/` instantly

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


## 🙏 Acknowledgments

- [Python Software Foundation](https://www.python.org/) for the official glossary
- [HuggingFace](https://huggingface.co/) for Sentence Transformers
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [OpenRouter](https://openrouter.ai/) for LLM API access
- [Meta AI](https://ai.meta.com/) for Llama 3.1

---

<p align="center">
  Made with ❤️ for Python learners everywhere
</p>
