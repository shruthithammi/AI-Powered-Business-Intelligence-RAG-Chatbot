# AI-Powered Business Intelligence RAG Chatbot

## About the Project

I built this project to make it easier to understand information in financial reports without having to go through the entire PDF manually.

The chatbot allows users to upload a financial report and ask questions in natural language. Instead of searching for information page by page, users can ask about revenue, expenses, net income, and other details, and the chatbot generates answers using the relevant content retrieved from the document.

I also developed another version that combines PDF-based question answering with web search. This helps when the uploaded document does not contain all the information needed to answer a question.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information before passing it to a language model to generate an answer.

## What I Built

The project contains two Streamlit applications:

**1. PDF-Based RAG Chatbot (`bot.py`)**

- Upload one or more PDF documents.
- Extract text from the uploaded PDFs.
- Split the extracted text into smaller chunks.
- Convert the chunks into embeddings and store them in a FAISS vector store.
- Ask questions about the uploaded documents.
- Retrieve the three most relevant chunks and use them as context for the language model's answer.

**2. PDF + Web Search Chatbot (`bot_websearch.py`)**

- Answer questions using uploaded PDF documents, web search, or both.
- Search the web using DuckDuckGo without requiring a separate search API key.
- Prefer information from the PDF for exact financial figures.
- Use web search results when the PDF does not contain the required information.
- Display available web source links when web search returns results.

## Technologies Used

- **Python** – Core programming language
- **Streamlit** – Web interface and chat functionality
- **LangChain** – PDF processing and text splitting
- **FAISS** – Vector storage and similarity search
- **Hugging Face Sentence Transformers** – Text embeddings using `paraphrase-MiniLM-L3-v2`
- **Ollama** – Local language model integration
- **Phi-3** – Default language model
- **PyPDF** – PDF document loading through LangChain
- **DuckDuckGo Search** – Web search in the hybrid version
- **Jupyter Notebook** – Step-by-step implementation and testing of the RAG pipeline

## How It Works

1. **Upload a PDF:** The user uploads a financial report through the Streamlit interface.
2. **Extract text:** The application reads the PDF and extracts its text.
3. **Split into chunks:** The text is divided into smaller sections of 500 characters, with an overlap of 50 characters.
4. **Generate embeddings:** Each chunk is converted into a numerical representation using a sentence-transformer model.
5. **Store and retrieve:** FAISS indexes the embeddings and retrieves the three most relevant chunks for a question.
6. **Generate an answer:** The retrieved content is passed to the language model along with instructions to answer using the available context.
7. **Combine with web search (hybrid version):** When enabled, DuckDuckGo results can provide additional information if the PDF does not contain the answer.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python
- Git
- Ollama
- VS Code or another Python editor

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Powered-Business-Intelligence-RAG-Chatbot.git
cd AI-Powered-Business-Intelligence-RAG-Chatbot
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the Required Packages

For the PDF-based chatbot:

```bash
pip install streamlit openai langchain langchain-community langchain-text-splitters pypdf faiss-cpu sentence-transformers
```

For the PDF + web search version, also install:

```bash
pip install duckduckgo-search
```

### 4. Set Up Ollama

Install Ollama and download the default model:

```bash
ollama pull phi3
```

Make sure Ollama is running locally and the model is available before starting the application.

The default configuration uses the local Ollama endpoint:

```text
http://localhost:11434/v1
```

The application also allows the base URL, API key, and model name to be changed from the sidebar.

### 5. Run the Application

To run the PDF-based chatbot:

```bash
streamlit run bot.py
```

To run the PDF + web search chatbot:

```bash
streamlit run bot_websearch.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser, upload a PDF, and start asking questions.

## Example Use Case

The repository includes a sample financial report containing Meta Platforms' fourth-quarter and full-year 2024 results.

After uploading the report, a user could ask questions such as:

- What was Meta's total revenue in 2024?
- What was its net income in 2024?
- How did revenue change compared with 2023?
- What were the company's capital expenditures?

The chatbot retrieves relevant sections of the report and uses them to prepare an answer. The figures should be checked against the original report when accuracy is important.

## Key Things I Learned

While working on this project, I explored how a RAG pipeline connects document processing, embeddings, vector search, and language model generation.

I also worked with PDF loading, text chunking, semantic similarity search, prompt construction, and integrating a local language model into a Streamlit application.

Building the web search version helped me understand how information from different sources can be combined to answer a user's question.

## Limitations

- Answers depend on the quality and content of the uploaded documents and retrieved web results.
- The chatbot may not answer correctly if relevant information is missing or not retrieved.
- Web search availability depends on the search service and network connection.
- The first run may take longer because the embedding model needs to be downloaded and loaded.
- Performance depends on the available system resources and the selected language model.

This project is intended as a practical exploration of RAG and document-based question answering, particularly for financial information.
