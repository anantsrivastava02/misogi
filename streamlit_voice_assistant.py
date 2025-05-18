import streamlit as st
import openai
import pinecone
import numpy as np
import soundfile as sf
import tempfile
import os
import time
from dotenv import load_dotenv
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

# Load environment variables
load_dotenv()
openai.api_key = OPENAI_API_KEY = ""
PINECONE_API_KEY = ""
PINECONE_ENVIRONMENT = "us-east1-gcp"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "scraped-content"


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
GPT_MODEL = "gpt-3.5-turbo"
OUTPUT_FILENAME = "response.wav"

# --- Pinecone Query Functions ---
@st.cache_resource
def initialize_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[chunk]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            embeddings.append([0] * EMBEDDING_DIMENSION)
    return embeddings

def query_similar_content(query_text, top_k=3):
    index = initialize_pinecone()
    query_embedding = get_embeddings([query_text])[0]
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    results = []
    for match in query_results.matches:
        results.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("url", match.metadata.get("source", "")),
            "title": match.metadata.get("title", "")
        })
    return results

def generate_answer(query, context_data):
    context = ""
    for i, item in enumerate(context_data):
        context += f"\nSOURCE {i+1} ({item['title']}):\n{item['text']}\n"
    messages = [
        {"role": "system", "content": (
            "You are a helpful voice assistant that provides informative answers based on retrieved content. "
            "Use the provided context to answer the user's question. "
            "If the context doesn't contain relevant information, say so politely. "
            "Keep answers concise and conversational, suitable for voice responses. "
            "Don't reference 'context' or 'sources' directly in your answer."
        )},
        {"role": "user", "content": f"CONTEXT: {context}\n\nQUESTION: {query}\n\nPlease provide a helpful answer:"}
    ]
    try:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I'm sorry, I wasn't able to generate an answer. Please try again."

def text_to_speech(text, output_file=OUTPUT_FILENAME):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except:
                with open(audio_file_path, "rb") as audio_file:
                    response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    text = response.text
            return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# --- Streamlit UI ---
st.set_page_config(page_title="Voice RAG Assistant", layout="centered")
st.title("🎤 Voice RAG Assistant")
st.write("Ask a question by recording your voice or typing below. The assistant will answer and speak back.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Audio recording with streamlit-webrtc
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()
        self.recorded_frames.append(pcm)
        return frame

audio_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    audio_processor_factory=AudioProcessor,
    video_transformer_factory=None,  # disables video
    async_processing=False,
)

st.write("Press Start, record your question, then Stop. Or type your question below.")
text_input = st.text_input("Or type your question:")

# Add a button to process the recorded audio after stopping
if st.button("Process Recorded Audio"):
    query = None
    if audio_ctx and hasattr(audio_ctx, "audio_processor") and audio_ctx.audio_processor:
        frames = audio_ctx.audio_processor.recorded_frames
        if frames:
            audio_data = np.concatenate(frames, axis=0)
            # Ensure audio_data is float32 for soundfile, then convert to int16 for PCM
            if audio_data.ndim == 1:
                audio_data = np.expand_dims(audio_data, axis=1)  # (samples, 1)
            # Normalize if needed (if values are not in int16 range)
            if audio_data.dtype != np.int16:
                # If float32, assume range -1.0 to 1.0, convert to int16
                if np.issubdtype(audio_data.dtype, np.floating):
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, 16000, subtype='PCM_16')
                tmp_path = tmp.name
            query = transcribe_audio(tmp_path)
            os.remove(tmp_path)
            st.write(f"**Transcribed:** {query}")
            st.session_state.history.append(("user", query))
            with st.spinner("Searching knowledge base..."):
                search_results = query_similar_content(query, top_k=3)
            with st.spinner("Generating answer..."):
                if not search_results:
                    answer = "I couldn't find any relevant information in my knowledge base."
                else:
                    answer = generate_answer(query, search_results)
            st.session_state.history.append(("assistant", answer))
            st.write(f"**Assistant:** {answer}")
            with st.spinner("Converting answer to speech..."):
                speech_file = text_to_speech(answer)
            if speech_file:
                st.audio(speech_file, format='audio/wav')
                os.remove(speech_file)
        else:
            st.warning("No audio recorded. Please record your question and then press this button.")

if st.button("Ask (Text)"):
    if text_input.strip():
        query = text_input.strip()
        st.session_state.history.append(("user", query))
        with st.spinner("Searching knowledge base..."):
            search_results = query_similar_content(query, top_k=3)
        with st.spinner("Generating answer..."):
            if not search_results:
                answer = "I couldn't find any relevant information in my knowledge base."
            else:
                answer = generate_answer(query, search_results)
        st.session_state.history.append(("assistant", answer))
        st.write(f"**Assistant:** {answer}")
        with st.spinner("Converting answer to speech..."):
            speech_file = text_to_speech(answer)
        if speech_file:
            st.audio(speech_file, format='audio/wav')
            os.remove(speech_file)
    else:
        st.warning("Please type your question.")

# Display chat history
st.markdown("---")
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")