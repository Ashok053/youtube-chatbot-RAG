import os
os.environ['GROQ_API_KEY'] = "add your api key here"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


import yt_dlp
import json

def get_clean_transcripts(url, lang="en"):
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": [lang]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get("automatic_captions") or info.get("subtitles")
        if not subs or lang not in subs:
            print(" No subtitles available")
            return []

        sub_url = subs[lang][0]["url"]

        import requests
        response = requests.get(sub_url)
        data = response.text.strip()
        if not data:
            print(" Subtitle file empty")
            return []

        try:
            json_data = json.loads(data)
        except json.JSONDecodeError:
            # YouTube JSON sometimes comes as multiple lines; fix it
            json_data = json.loads("{" + data.split("{", 1)[1])

        results = []
        for event in json_data.get("events", []):
            if "segs" not in event:
                continue
            start = event.get("tStartMs", 0) / 1000.0
            duration = event.get("dDurationMs", 0) / 1000.0
            text = "".join(seg.get("utf8", "") for seg in event["segs"]).replace("\n", " ").strip()
            if text:
                results.append({
                    "text": text,
                    "start": round(start, 3),
                    "duration": round(duration, 3)
                })
        return results


def fetch_transcript(url):
    transcript = get_clean_transcripts(url)
    if not transcript:
        return None
    combined_text = " ".join(chunk["text"] for chunk in transcript)
    return combined_text


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    return chunks



llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.2,
    max_tokens=512
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant. Answer only from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
    """,
    input_variables=['context','question']
)



from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def create_main_chain(retriever):
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


import streamlit as st

st.title("YouTube Video Chatbot 🎥🤖")

url = st.text_input("Paste YouTube Video URL:")

if url:
    with st.spinner("Fetching transcript..."):
        combined_text = fetch_transcript(url)
    if combined_text:
        st.success("Transcript fetched!")

        # Step 1: Indexing
        retriever = create_vector_store(combined_text)

        # Step 2: Create chain
        main_chain = create_main_chain(retriever)

        st.info("Ask anything about this video:")
        query = st.text_input("Your question:")
        if query:
            with st.spinner("Generating answer..."):
                answer = main_chain.invoke(query)
            st.write(answer)
    else:
        st.error("Transcript not available for this video.")
