import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import librosa
import os
import json
from datetime import datetime
import logging
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="Women Safety Analysis System",
    page_icon="🚺",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stVideo {width: 100%; height: auto;}
    .css-1y4p8pa {padding-top: 0rem;}
    .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

class WomenSafetyAnalyzer:
    def __init__(self):
        genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_video_safety(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frames = []
            frame_count = 0
            safety_concerns = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % int(fps) == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    frames.append(pil_frame)
                    
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing frame {frame_count}/{total_frames}")
                    
                    if len(frames) >= 5:
                        safety_prompt = """
                        Analyze these frames for women's safety concerns, focusing on:
                        • Unwanted physical contact or assault
                        • Sexual harassment or inappropriate behavior
                        • Stalking or following behavior
                        • Threatening gestures or postures
                        • Signs of distress or fear
                        • Group intimidation
                        • Workplace harassment
                        • Public transport safety issues
                        • Domestic violence indicators
                        • Unsafe environment conditions
                        
                        Provide a clear assessment of:
                        1. Type of safety threat detected
                        2. Severity level of the concern
                        3. Immediate action recommendations
                        """
                        
                        chat = self.model.start_chat(history=[])
                        response = chat.send_message([safety_prompt, *frames])
                        safety_concerns.append(response.text)
                        frames = []
            
            cap.release()
            return safety_concerns
            
        except Exception as e:
            self.logger.error(f"Error in video analysis: {e}")
            return None

    def analyze_audio_safety(self, audio_path):
        try:
            y, sr = librosa.load(audio_path)
            features = {
                'duration': float(len(y) / sr),
                'rms_energy': float(librosa.feature.rms(y=y).mean()),
                'spectral_centroid': float(librosa.feature.spectral_centroid(y=y).mean()),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean())
            }
            
            audio_prompt = f"""
            Analyze these audio characteristics for signs of distress or safety concerns:
            Duration: {features['duration']:.2f} seconds
            Energy Level: {features['rms_energy']:.4f}
            
            Focus on detecting:
            • Distressed vocals or crying
            • Verbal harassment or threats
            • Calls for help
            • Aggressive or threatening tones
            • Signs of struggle or distress
            • Verbal abuse indicators
            • Emergency situations
            
            Provide analysis of:
            1. Type of audio threat detected
            2. Urgency level of the situation
            3. Recommended response actions
            """
            
            chat = self.model.start_chat(history=[])
            response = chat.send_message(audio_prompt)
            return response.text
        
        except Exception as e:
            self.logger.error(f"Error in audio analysis: {e}")
            return None

    def analyze_content(self, video_file, audio_file):
        results = {
            "timestamp": datetime.now().isoformat(),
            "video_analysis": None,
            "audio_analysis": None
        }

        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                results["video_analysis"] = self.analyze_video_safety(tmp_video.name)
            os.unlink(tmp_video.name)

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                tmp_audio.write(audio_file.read())
                results["audio_analysis"] = self.analyze_audio_safety(tmp_audio.name)
            os.unlink(tmp_audio.name)

        return results

def main():
    # Display header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #FF69B4, #FFB6C1); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h1 style='color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>Women Safety Analysis System</h1>
            <p style='color: white; font-size: 1.2rem;'>Empowering Safety Through Technology</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### Analysis Features
        • Real-time safety detection
        • Harassment recognition
        • Distress signal analysis
        • Threat assessment
        • Safety recommendations
        """)

    # Main content
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.05); border-radius: 10px; padding: 1rem; margin-bottom: 1rem;'>
                <div style='color: #FF69B4; font-size: 1.1rem; font-weight: 500;'>📹 Video Analysis</div>
                <div style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>Upload MP4, AVI, MOV (Max 200MB)</div>
            </div>
        """, unsafe_allow_html=True)
        video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")

        st.markdown("""
            <div style='background: rgba(255,255,255,0.05); border-radius: 10px; padding: 1rem; margin-bottom: 1rem;'>
                <div style='color: #FF69B4; font-size: 1.1rem; font-weight: 500;'>🎤 Audio Analysis</div>
                <div style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>Upload WAV, MP3 (Max 200MB)</div>
            </div>
        """, unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3'], label_visibility="collapsed")

        if video_file:
            st.video(video_file)
        if audio_file:
            st.audio(audio_file)

    with col2:
        st.subheader("Analysis Results")
        analyze_button = st.button("Begin Safety Analysis", 
                                type="primary",
                                use_container_width=True,
                                disabled=(not video_file and not audio_file))

        if analyze_button:
            analyzer = WomenSafetyAnalyzer()
            with st.spinner("Analyzing content for safety concerns..."):
                results = analyzer.analyze_content(video_file, audio_file)
                
                if results.get("video_analysis") or results.get("audio_analysis"):
                    st.markdown("""
                        <div style='background: linear-gradient(45deg, #2b1f47, #1a1a2e); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <h3 style='color: #FF69B4; margin-bottom: 1rem;'>Safety Assessment</h3>
                            <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;'>
                    """, unsafe_allow_html=True)
                    
                    if results.get("video_analysis"):
                        analysis_text = results["video_analysis"][0] if results["video_analysis"] else ""
                        st.markdown(f"<div style='color: white;'>{analysis_text}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download Safety Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_report"
                    )
                else:
                    st.error("Analysis failed or no safety concerns detected.")

if __name__ == "__main__":
    main()
