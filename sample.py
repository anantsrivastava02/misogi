import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import os
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️")

# Function to record audio
def record_audio(duration=5, sample_rate=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished")
    return recording, sample_rate

# Function to save audio
def save_audio(recording, sample_rate, filename):
    # Ensure the recordings directory exists
    os.makedirs("recordings", exist_ok=True)
    # Save the recording as a WAV file
    file_path = os.path.join("recordings", filename)
    wavfile.write(file_path, sample_rate, recording)
    return file_path

# Main Streamlit app
def main():
    st.title("Audio Recorder App 🎙️")
    st.write("Click the button to start recording a 5-second audio clip.")

    # Record button
    if st.button("Record Audio"):
        # Record 5 seconds of audio
        recording, sample_rate = record_audio(duration=5)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        
        # Save the audio
        file_path = save_audio(recording, sample_rate, filename)
        
        # Display success message and audio player
        st.success(f"Audio saved as {file_path}")
        st.audio(file_path)

        # Provide download button
        with open(file_path, "rb") as file:
            st.download_button(
                label="Download Audio",
                data=file,
                file_name=filename,
                mime="audio/wav"
            )

if __name__ == "__main__":
    main()