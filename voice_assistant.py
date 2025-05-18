import os
import time
import openai
import pinecone
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import tempfile
from typing import List
from dotenv import load_dotenv
import speech_recognition as sr

# Load environment variables
load_dotenv()
openai.api_key = OPENAI_API_KEY = "s"
PINECONE_API_KEY = "pr"
PINECONE_ENVIRONMENT = "us-east1-gcp"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "scraped-content"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
GPT_MODEL = "gpt-3.5-turbo"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
OUTPUT_FILENAME = "response.wav"

# --- Pinecone Query Functions ---
def initialize_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

def get_embeddings(chunks: List[str]) -> List[List[float]]:
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
            print(f"Error getting embeddings: {e}")
            embeddings.append([0] * EMBEDDING_DIMENSION)
    return embeddings

def query_similar_content(query_text: str, top_k: int = 5):
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

# --- Voice and TTS Functions ---
def record_audio(duration=RECORD_SECONDS):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    for _ in range(duration):
        print(".", end="", flush=True)
        time.sleep(1)
    print("\nRecording complete!")
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_filename = temp_file.name
    temp_file.close()
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    return temp_filename

def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            print("Transcribing...")
            try:
                text = recognizer.recognize_google(audio_data)
            except:
                with open(audio_file_path, "rb") as audio_file:
                    response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    text = response.text
            print(f"Transcription: {text}")
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""
    finally:
        try:
            os.remove(audio_file_path)
        except:
            pass

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
        print(f"Error generating answer: {e}")
        return "I'm sorry, I wasn't able to generate an answer. Please try again."

def text_to_speech(text, output_file=OUTPUT_FILENAME):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(output_file)
        print(f"Speech saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None

def play_audio(file_path):
    try:
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        status = sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def voice_assistant():
    print("Voice RAG Assistant Starting...")
    print("Make sure your Pinecone index has been populated with your scraped content.")
    while True:
        print("\n" + "="*50)
        print("Press Enter to ask a question (or type 'exit' to quit)")
        user_input = input()
        if user_input.lower() == 'exit':
            print("Thank you for using Voice RAG Assistant. Goodbye!")
            break
        try:
            audio_file = record_audio()
            query = speech_to_text(audio_file)
            if not query:
                print("Sorry, I couldn't understand that. Please try again.")
                continue
            print(f"Processing question: {query}")
            print("Searching knowledge base...")
            search_results = query_similar_content(query, top_k=3)
            if not search_results:
                answer = "I couldn't find any relevant information in my knowledge base."
            else:
                print("Generating answer...")
                answer = generate_answer(query, search_results)
            print(f"Answer: {answer}")
            print("Converting answer to speech...")
            speech_file = text_to_speech(answer)
            if speech_file:
                print("Playing response...")
                play_audio(speech_file)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    voice_assistant() 