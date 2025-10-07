# 🎬 YouTube Chatbot RAG (Groq + LangChain)

A **Streamlit-based application** that allows users to **chat with any YouTube video** using Retrieval-Augmented Generation (RAG).  
Just paste a YouTube link, and the app fetches transcripts, embeds them, and answers questions about the video content.

---

## 🚀 Features
- 📥 Automatically fetches and cleans YouTube transcripts  
- 💡 Splits text into chunks and generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`  
- 🧠 Stores embeddings in FAISS for vector search  
- 🤖 Powered by Groq LLM (`meta-llama/llama-4-maverick-17b`) for context-aware answers  
- 💬 Interactive chat interface built using Streamlit  
---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ashok053/youtube-chatbot-RAG.git
cd youtube-chatbot-RAG
````
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Set Up Your Groq API Key
Create a free API key from Groq Console: https://console.groq.com/keys
Then set it as an environment variable:
```bash
export GROQ_API_KEY="your_api_key_here"
```
### Run as 
```bash
streamlit run main.py
```

## 📦 Tech Stack
- LangChain
- FAISS
- Groq API (Llama-4-Maverick)
- HuggingFace Embeddings
- Streamlit UI
- yt_dlp for YouTube data fetching
