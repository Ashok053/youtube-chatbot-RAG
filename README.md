🎥 YouTube Video Chatbot (RAG + Groq + LangChain)

This project lets you chat with any YouTube video — just paste the video link, and it extracts the transcript, embeds it into a vector store (FAISS), and answers your questions using Groq’s LLaMA model


🚀 Features

🧩 Extracts subtitles or auto-captions from YouTube
✂️ Splits and embeds transcript into vector form using LangChain + FAISS
🧠 Uses Groq LLM (LLaMA-4 model : meta-llama/llama-4-maverick-17b-128e-instruct ) for lightning-fast reasoning
💬 Ask any question about the video content
🪄 Built with Streamlit (UI) + Python (LangChain)



🛠️ Tech Stack

LangChain (RAG framework)
Groq API (LLM inference)
FAISS (Vector database)
HuggingFace Embeddings
yt_dlp (YouTube transcript extraction)
Streamlit (Frontend UI)



⚙️ Setup Instructions
1️⃣ Clone this repository
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Set up your Groq API Key
Create your free API key at 👉 https://console.groq.com/keys
Then add it to your environment:
export GROQ_API_KEY="your_api_key_here"

Or in Windows PowerShell:
setx GROQ_API_KEY "your_api_key_here"

▶️ Run the Streamlit app
streamlit run main.py


Then open your browser at:
http://localhost:8501
