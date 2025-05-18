# Misogi Voice RAG Assistant

This repository provides a Voice Retrieval-Augmented Generation (RAG) Assistant that leverages OpenAI and Pinecone to answer questions based on ingested knowledge. It supports both web (Streamlit) and terminal-based voice interactions.

## Features
- **Voice-based Q&A**: Ask questions by voice and get spoken answers.
- **Contextual Search**: Uses Pinecone vector search for relevant context.
- **OpenAI GPT Integration**: Generates concise, conversational answers.
- **Text-to-Speech**: Answers are spoken back to the user.
- **Flexible Interfaces**: Use via web (Streamlit) or terminal.

## Main Scripts

### 1. `Final1.py`
- **Streamlit web app** for voice Q&A.
- **Does NOT delete chat history** between questions (session persists).

### 2. `Final2.py`
- **Streamlit web app** for voice Q&A.
- **Deletes chat history** after each question (session resets).

### 3. `ingest.py`
- **Ingests data into Pinecone**.
- Scrapes and processes content, generates embeddings, and uploads them to your Pinecone index.

### 4. `voice_assistant.py`
- **Runs the voice assistant in the terminal** (no web interface).
- Record your question, get an answer, and hear it spoken back.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Misogi
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=us-east1-gcp
   PINECONE_INDEX_NAME=scraped-content
   ```

## Usage

### Ingest Data
Before asking questions, ingest your data into Pinecone:
```bash
python ingest.py
```

### Web Voice Assistant
- **With persistent chat history:**
  ```bash
  streamlit run Final1.py
  ```
- **With chat history deleted after each question:**
  ```bash
  streamlit run Final2.py
  ```

### Terminal Voice Assistant
```bash
python voice_assistant.py
```

## Notes
- Ensure your Pinecone index is populated before using the assistant.
- The assistant uses OpenAI for embeddings, chat, and text-to-speech.
- For best results, use high-quality audio input.

## Requirements
See `requirement.txt` for all dependencies.

## License
MIT (or specify your license here) 