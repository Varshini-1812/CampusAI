
# CampusAI – Agentic University Student Support Assistant

CampusAI is an AI-powered university student support assistant that answers questions about university information using Retrieval-Augmented Generation (RAG), analyzes student feedback, and generates structured feedback reports.

The application uses a locally hosted Qwen model through Ollama for university question answering and agent-based tool selection. Student feedback analysis currently uses rule-based keyword matching.

## Features

### 1. University Information Assistant
- Answers questions using university documents.
- Retrieves relevant information from PDF documents.
- Uses semantic search with MiniLM embeddings.
- Stores and retrieves document embeddings using ChromaDB.
- Uses Qwen through Ollama to generate responses.

### 2. Agent-Based Tool Selection
- Uses LangChain tools for different tasks.
- Selects relevant tools based on user requests.
- Supports university information retrieval and student feedback operations.

### 3. Student Feedback Analysis
- Identifies positive feedback and common concerns.
- Classifies comments using keyword-based rules.
- Displays matching comments as evidence.
- Highlights unclassified feedback.

### 4. Feedback Report Generation
- Generates structured student feedback reports.
- Includes positive themes and areas of concern.
- Provides suggested actions based on identified themes.
- Explains dataset limitations.

## Tech Stack

- **Language:** Python
- **LLM:** Qwen 2.5 via Ollama
- **Agent Framework:** LangChain
- **RAG:** Retrieval-Augmented Generation
- **Embeddings:** Sentence Transformers (MiniLM)
- **Vector Database:** ChromaDB
- **PDF Processing:** PyPDF
- **Frontend:** Streamlit
- **Data Processing:** Pandas / Python

## Project Structure

```text
CampusAI/
├── data/
│   ├── pdfs/
│   ├── student_feedback.csv
│   └── university_info.txt
├── src/
├── tests/
├── agent.py
├── agent_tools.py
├── app.py
├── feedback_tools.py
├── ingest.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Varshini-1812/CampusAI.git
cd CampusAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama and download the model used by the project.

```bash
ollama pull qwen2.5:7b
```

Start Ollama before running the application.

### 5. Prepare the knowledge base

Run the ingestion script:

```bash
python ingest.py
```

This processes the university documents and creates the ChromaDB vector database.

### 6. Launch the application

```bash
streamlit run app.py
```

## Current Limitations

- Student feedback classification uses keyword matching rather than LLM-based semantic analysis.
- Feedback results depend on the available dataset and configured keywords.
- The quality of university answers depends on the documents available in the knowledge base.
- The local LLM requires Ollama and sufficient system resources.

## Future Improvements

- Improve feedback classification using semantic analysis.
- Expand the university knowledge base.
- Improve handling of questions outside the available documents.
- Add more comprehensive testing and evaluation.

## Author

Varshini Soogoori

