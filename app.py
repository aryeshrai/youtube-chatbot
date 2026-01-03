import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")

st.title("🤖 Chat with YouTube Videos (RAG)")
st.markdown("Enter a YouTube URL and your Groq API Key to chat with the video's content.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    youtube_url = st.text_input("Enter YouTube Video URL")
    process_button = st.button("Process Video")
    
    st.markdown("---")
    st.markdown("**Note:** This app uses `llama-3.3-70b-versatile` via Groq and `all-MiniLM-L6-v2` for embeddings.")

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper Functions ---

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once to improve performance."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_video_id(url):
    """Extract video ID from URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

def process_video(url):
    video_id = get_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        return None

    try:
        with st.spinner("Fetching transcript..."):
            api = YouTubeTranscriptApi()
            transcript_data = api.fetch(video_id, languages=['en', 'hi'])

            # IMPORTANT FIX 👇
            transcript_text = " ".join(
                snippet.text for snippet in transcript_data
            )

        with st.spinner("Splitting text and creating embeddings..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.create_documents([transcript_text])

            embeddings = load_embedding_model()
            vector_store = FAISS.from_documents(chunks, embeddings)

        st.success("Video processed successfully!")
        return vector_store

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_rag_chain(vector_store, api_key):
    """Creates the RAG chain."""
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    
    Question: {question}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Main Logic ---

# 1. Processing Video
if process_button:
    if not groq_api_key:
        st.warning("Please enter your Groq API Key.")
    elif not youtube_url:
        st.warning("Please enter a YouTube URL.")
    else:
        st.session_state.vector_store = process_video(youtube_url)
        # Clear chat history when new video is loaded
        st.session_state.chat_history = []

# 2. Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handling User Input
if prompt := st.chat_input("Ask a question about the video..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    if st.session_state.vector_store is None:
        st.error("Please process a video first.")
    elif not groq_api_key:
        st.error("Please enter your Groq API Key.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chain = get_rag_chain(st.session_state.vector_store, groq_api_key)
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")