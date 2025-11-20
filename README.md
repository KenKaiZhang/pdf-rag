# PDF RAG System

This project is a simple Retrieval-Augmented Generation (RAG) system designed to answer questions about PDF documents. It provides a web-based interface using Streamlit to upload a PDF and ask questions about its content.

This project was created as a learning exercise to understand the fundamentals of building a RAG pipeline from scratch.

## Features

-   **Web Interface:** Easy-to-use interface built with Streamlit.
-   **PDF Upload:** Upload your own PDF files for analysis.
-   **Local LLMs:** Powered by local models via Ollama, ensuring privacy and no API key costs.
-   **Vector Storage:** Uses ChromaDB to store document embeddings.
-   **Source Verification:** Displays the source chunks from the document that were used to generate the answer.

## How It Works

The application follows a standard RAG pipeline:

1.  **Upload:** A user uploads a PDF file through the Streamlit interface.
2.  **Processing:** The PDF is processed to extract text and tabular data.
3.  **Chunking:** The extracted content is divided into smaller, manageable chunks.
4.  **Embedding:** Each chunk is converted into a numerical representation (embedding) using the `nomic-embed-text` model running in Ollama.
5.  **Storage:** The embeddings are stored in a persistent ChromaDB vector store.
6.  **Retrieval:** When a user asks a question, the query is embedded, and the system retrieves the most relevant chunks from ChromaDB based on vector similarity.
7.  **Generation:** The user's question and the retrieved context are passed to a Large Language Model (`llama3.2` via Ollama) to generate a final answer.

## Tech Stack

-   **Frontend:** Streamlit
-   **Orchestration:** LangChain
-   **LLM & Embeddings:** Ollama (`llama3.2`, `nomic-embed-text`)
-   **Vector Store:** ChromaDB
-   **PDF Parsing:** `PyMuPDF`

## Getting Started

### Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
-   [Ollama](https://ollama.com/) installed and running on your host machine.

### 1. Set up Ollama

You need to have Ollama running with the required models.

1.  Install [Ollama](https://ollama.com/).
2.  Pull the required models:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```
3.  Ensure the Ollama server is running.

### 2. Run the Application

There are two ways to run the application:

#### Option A: Docker Compose (Recommended)

This is the easiest way to get started. It will run the Streamlit application and the ChromaDB database in containers.

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd pdf-rag
    ```
2.  Build and start the services:
    ```bash
    docker-compose up --build
    ```
    *Note: The `docker-compose.yaml` in this repository only defines the `chromadb` service. You will need to add the `app` service to it to run the full application with a single command.*

3.  Access the application in your browser at `http://localhost:8501`.

#### Option B: Manual Local Setup

1.  **Start ChromaDB:**
    Use Docker to start the ChromaDB service.
    ```bash
    docker-compose up -d chromadb
    ```

2.  **Set up Python Environment:**
    ```bash
    # Clone the repository
    git clone <your-repository-url>
    cd pdf-rag

    # Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -e .
    ```

3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

4.  Access the application in your browser at `http://localhost:8501`.

## Configuration

The application can be configured by editing the `src/configs/defaults.py` file. Here you can change:
-   Ollama model names (`MODEL_NAME`, `EMBEDDING_MODEL`)
-   Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
-   Retrieval parameters (`TOP_K_RESULTS`)

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── docker-compose.yaml     # Docker Compose configuration for ChromaDB
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file
├── chroma_data/            # Directory for persistent ChromaDB data
└── src/
    ├── configs/
    │   └── defaults.py     # Default configuration settings
    └── helpers/
        ├── pdf_processor.py      # Logic for processing and chunking PDFs
        └── vector_store_manager.py # Manages the ChromaDB vector store
```
