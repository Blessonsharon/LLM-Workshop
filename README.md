# 🎵 Creative AI Co-Writer — Music Industry LLM

An AI-powered creative assistant for musicians, built on **Google Gemini** with **Spotify** integration and a **RAG (Retrieval-Augmented Generation)** pipeline. It acts as a virtual multi-platinum music producer, songwriter, and co-writer right in your terminal.

---

## ✨ Features

### 🤖 AI Co-Writer Chat (`co-write.py`)

| Feature | Description |
|---|---|
| **Lyric Generation & Refinement** | Suggests rhymes, metaphors, and structural improvements for verses, choruses, and bridges. Explains *why* a line works rhythmically. |
| **Chord Progression Recommendations** | Generates culturally relevant, genre-specific chord progressions with both standard chord names and Roman numeral notation. |
| **Production Advice** | Provides specific tips on instrumentation, arrangement, mixing, and sound design. |
| **Gemini 2.5 Flash Model** | Powered by `gemini-2.5-flash` for fast, high-quality creative responses. |
| **Configurable Temperature** | Set to `0.7` for a balance between creativity and coherence. |
| **Persistent Chat Session** | Maintains conversation context across turns so the AI remembers your entire session. |
| **Automatic Retry with Back-off** | Handles Gemini API rate limits (HTTP 429) gracefully with progressive retry delays (15s → 30s → 45s). |

### 🎧 Spotify Integration (Optional)

| Feature | Description |
|---|---|
| **Track Search** | Search any song by name and optionally by artist for accurate results. |
| **Audio Feature Analysis** | Retrieves detailed audio features for any track: **Tempo (BPM)**, **Musical Key & Mode** (Major/Minor), **Danceability**, **Energy**, **Time Signature**. |
| **Automatic Function Calling** | Gemini automatically invokes the Spotify tool when relevant — no manual commands needed. Just ask *"What key is Blinding Lights?"* |
| **Graceful Degradation** | Spotify is fully optional; the co-writer works perfectly without it. |

### 📚 RAG Pipeline (`rag_setup.py`)

| Feature | Description |
|---|---|
| **Document Ingestion** | Loads `.txt` and `.pdf` files from a local `knowledge_base/` directory. |
| **Text Chunking** | Splits documents using `RecursiveCharacterTextSplitter` (500 chars, 50 char overlap) for optimal retrieval. |
| **Embeddings** | Generates vector embeddings via Google's `gemini-embedding-001` model. |
| **FAISS Vector Store** | Stores and retrieves embeddings locally using Facebook's FAISS for fast similarity search. |
| **RAG-Enhanced Chat** | Answers questions using retrieved context from your custom knowledge base, prioritizing your documents over general knowledge. |
| **Music Producer Persona** | The RAG chat retains the same expert co-writer persona as the main chat. |

### 🛡️ Robustness & UX

| Feature | Description |
|---|---|
| **Environment Variable Config** | API keys (`GEMINI_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`) are loaded from environment variables with sensible defaults. |
| **Clean Error Handling** | Catches and reports errors gracefully without crashing the session. |
| **Windows Terminal Compatible** | Uses ASCII-safe sharp/flat symbols (e.g., `C#/Db`) for broad terminal compatibility. |
| **Keyboard Interrupt Support** | Gracefully exits on `Ctrl+C`. |

---

## 🚀 Quick Start

### 1. Run the AI Co-Writer
```bash
python co-write.py
```

### 2. Ingest Documents into the RAG Knowledge Base
```bash
python rag_setup.py ingest
```

### 3. Chat with RAG-Enhanced Co-Writer
```bash
python rag_setup.py chat
```

---

## 📁 Project Structure

```
LLM Workshop/
├── co-write.py          # Main AI co-writer with Spotify + retry logic
├── co_write.py          # Earlier version of the co-writer
├── rag_setup.py         # RAG pipeline (ingest & chat)
├── knowledge_base/      # Drop .txt and .pdf files here for RAG
│   └── sample_music_theory.txt
├── faiss_index/         # Auto-generated FAISS vector store
└── README.md            # This file
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |
| `SPOTIFY_CLIENT_ID` | No | Spotify Developer app client ID |
| `SPOTIFY_CLIENT_SECRET` | No | Spotify Developer app client secret |

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini (`gemini-2.5-flash` / `gemini-1.5-flash`)
- **Spotify API**: `spotipy` library
- **RAG Framework**: LangChain + FAISS
- **Embeddings**: `gemini-embedding-001`
- **Language**: Python 3
