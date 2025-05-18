import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wavfile
import openai
import pinecone
import numpy as np
import os
import atexit
import logging
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")  # Replace with your key if not using .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")  # Replace with your key if not using .env
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "scraped-content")

openai.api_key = OPENAI_API_KEY

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
GPT_MODEL = "gpt-3.5-turbo"
OUTPUT_FILENAME = "response.wav"
SAMPLE_RATE = 44100
RECORD_DURATION = 5  # seconds

# Directory for recordings
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Function to clean up audio files
def cleanup_audio_files():
    try:
        for file in os.listdir(RECORDINGS_DIR):
            file_path = os.path.join(RECORDINGS_DIR, file)
            if file.endswith(".wav"):
                os.remove(file_path)
                logger.info(f"Deleted audio file: {file_path}")
        if os.path.exists(OUTPUT_FILENAME):
            os.remove(OUTPUT_FILENAME)
            logger.info(f"Deleted response audio: {OUTPUT_FILENAME}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function to run on exit
atexit.register(cleanup_audio_files)

# Function to record audio
def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording finished")
    return recording, sample_rate

# Function to save audio
def save_audio(recording, sample_rate, filename):
    file_path = os.path.join(RECORDINGS_DIR, filename)
    wavfile.write(file_path, sample_rate, recording)
    return file_path

# Pinecone initialization
@st.cache_resource
def initialize_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)
# Get embeddings for text
def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[chunk.replace("\n", " ")]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            logger.error(f"Embedding error: {e}", exc_info=True)
            embeddings.append([0.0] * EMBEDDING_DIMENSION)
    return embeddings

# Query Pinecone for similar content
def query_similar_content(query_text, top_k=3):
    index = initialize_pinecone()
    query_embedding = get_embeddings([query_text])[0]
    if sum(abs(x) for x in query_embedding) == 0.0:
        st.warning("Could not generate embedding for the query.")
        return []
    try:
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("url", match.metadata.get("source", "")),
                "title": match.metadata.get("title", "")
            }
            for match in query_results.matches
        ]
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        logger.error(f"Pinecone query error: {e}", exc_info=True)
        return []

# Generate answer using OpenAI
def generate_answer(query, context_data):
    if not context_data:
        return "I couldn't find any relevant information to answer that question."
    context = ""
    for i, item in enumerate(context_data):
        context += f"\nSOURCE {i+1} (Title: {item.get('title', 'N/A')} - Source: {item.get('source', 'N/A')} - Score: {item.get('score', 0.0):.2f}):\n{item.get('text', '')}\n"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Provide informative answers based on the retrieved content. "
                "Keep answers concise and conversational, suitable for voice responses. "
                "Do not reference 'context' or 'sources' in your spoken answer."
            )
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nBased on the context, provide a helpful and concise answer."
        }
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
        logger.error(f"OpenAI chat error: {e}", exc_info=True)
        return "I'm sorry, I encountered an issue while generating an answer."

# Convert text to speech
def text_to_speech(text, output_file=OUTPUT_FILENAME):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(output_file)
        logger.info(f"Text-to-speech saved to {output_file}")
        return output_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        logger.error(f"OpenAI TTS error: {e}", exc_info=True)
        return None

# Transcribe audio
def transcribe_audio(audio_file_path):
    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) <= 44:
        logger.error(f"Audio file is missing or empty: {audio_file_path}")
        st.error("The recorded audio file seems to be empty or invalid.")
        return ""
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        logger.info(f"Audio transcribed: {response[:100]}...")
        return response
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Whisper transcription error: {e}", exc_info=True)
        return ""

# Main Streamlit app
def main():
    st.set_page_config(page_title="Voice RAG Assistant", layout="centered")
    st.title("🎤 Voice RAG Assistant")
    st.write("Record a 5-second audio question or type below. The assistant will answer and speak back.")
    st.info("Click 'Record Audio' to start, speak, then wait for the recording to finish.")

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Record audio button
    if st.button("Record Audio"):
        # Record and save audio
        recording, sample_rate = record_audio()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        audio_file_path = save_audio(recording, sample_rate, filename)

        # Transcribe audio
        transcription = transcribe_audio(audio_file_path)
        if transcription:
            st.write(f"**Transcription**: {transcription}")
            # Query Pinecone and generate answer
            context_data = query_similar_content(transcription)
            answer = generate_answer(transcription, context_data)
            st.write(f"**Answer**: {answer}")
            st.session_state.history.append({"question": transcription, "answer": answer})

            # Convert answer to speech
            audio_output = text_to_speech(answer)
            if audio_output:
                st.audio(audio_output)
                with open(audio_output, "rb") as file:
                    st.download_button(
                        label="Download Response Audio",
                        data=file,
                        file_name=OUTPUT_FILENAME,
                        mime="audio/wav"
                    )

    # Text input as an alternative
    query = st.text_input("Or type your question here:")
    if query:
        context_data = query_similar_content(query)
        answer = generate_answer(query, context_data)
        st.write(f"**Answer**: {answer}")
        st.session_state.history.append({"question": query, "answer": answer})
        audio_output = text_to_speech(answer)
        if audio_output:
            st.audio(audio_output)
            with open(audio_output, "rb") as file:
                st.download_button(
                    label="Download Response Audio",
                    data=file,
                    file_name=OUTPUT_FILENAME,
                    mime="audio/wav"
                )

    # Display chat history
    if st.session_state.history:
        st.write("### Chat History")
        for i, entry in enumerate(st.session_state.history):
            st.write(f"**Q{i+1}**: {entry['question']}")
            st.write(f"**A{i+1}**: {entry['answer']}")

if __name__ == "__main__":
    main()