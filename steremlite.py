import streamlit as st
from tempfile import NamedTemporaryFile
import moviepy.editor as mp
import time
from predict_realtime import VideoDescriptionRealTime
import config
from predict_realtime import translate_to_hindi, text_to_speech

# Dark Mode Custom CSS for Styling
st.markdown(
    """
    <style>
    /* Dark background and text colors */
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #f1f1f1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
    }
    /* Custom button styles */
    .stButton button {
        background-color: #333333 !important;
        color: #f1f1f1 !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        margin-top: 10px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #555555 !important;
        color: #e0e0e0 !important;
    }
    /* Side-by-side buttons for download and reset */
    .button-row {
        display: flex;
        justify-content: center;
        gap: 15px;
    }
    /* Video Information styling */
    .video-info {
        background-color: #292929;
        color: #d0d0d0;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: left;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title and Description
st.title("🎥 Non-Audio Video Summarization to Audio 🎧")
st.markdown("Upload a video, and our AI model will summarize it and convert it into audio format!")

# Instructions Section
st.markdown(
    """
    ### Steps:
    1. **Upload a video** (supported formats: mp4, avi, mov, mkv).
    2. The AI model generates a text summary and converts it into an audio file.
    3. **Download** and enjoy your summarized audio file!
    """
)

# Upload Video File
video_file = st.file_uploader("Upload your video:", type=["mp4", "avi", "mov", "mkv"])

# Columns for model, search type, and language selection
col1, col2, col3 = st.columns(3)
with col1:
    model_option = st.selectbox("Model:", ("LSTM", "LSTM + GRU", "GRU"))
with col2:
    search_type = st.selectbox("Search Type:", ("greedy", "beam"))
with col3:
    model_lang = st.selectbox("Language:", ("Hindi", "Marathi", "Kannada"))

# Process Video Button
if video_file:
    st.video(video_file)
    st.markdown("### Processing... Please wait.")
    progress_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent_complete + 1)
    
    # Save uploaded video to temporary file
    temp_video = NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()

    # Run Model Prediction and Translation
    video_to_text = VideoDescriptionRealTime(config, model_option, search_type)
    sentence = video_to_text.main_test(temp_video.name)
    print(sentence)
    hindi_text = translate_to_hindi(sentence, model_lang)
    
    # Convert text to audio
    audio_file_name = "summarized_audio.mp3"
    text_to_speech(hindi_text, audio_file_name, model_lang)
    
    st.audio(audio_file_name)

    # Button Row for Download and Reset
    st.markdown("<div class='button-row'>", unsafe_allow_html=True)
    st.download_button("📥 Download Audio Summary", data=open(audio_file_name, "rb").read(), 
                       file_name="summarized_audio.mp3", mime="audio/mp3")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Video Information
    video = mp.VideoFileClip(temp_video.name)
    st.markdown(
        f"""
        <div class='video-info'>
        <b>Video Information:</b><br>
        - **Duration**: {video.duration:.2f} seconds<br>
        - **Resolution**: {video.size[0]}x{video.size[1]} pixels<br>
        - **FPS**: {video.fps} frames per second
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("**Developed by Bhavesh**", unsafe_allow_html=True)
