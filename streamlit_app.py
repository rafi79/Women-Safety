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

# Must be the first Streamlit command
st.set_page_config(
    page_title="Content Analysis System",
    page_icon="📊",
    layout="wide"
)

class ContentAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
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
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_content(self, video_file, audio_file):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "video_analysis": None,
                "audio_analysis": None
            }

            if video_file is not None:
                self.logger.info("Analyzing video...")
                progress_bar.progress(25)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(video_file.read())
                    video_analysis = self.analyze_video(tmp_video.name)
                    results["video_analysis"] = video_analysis
                os.unlink(tmp_video.name)
                progress_bar.progress(75)

            if audio_file is not None:
                self.logger.info("Analyzing audio...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    audio_analysis = self.analyze_audio(tmp_audio.name)
                    results["audio_analysis"] = audio_analysis
                os.unlink(tmp_audio.name)

            self.save_results(results)
            return results

        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return {"error": str(e)}

    def analyze_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            frames_container = st.empty()
            analysis_container = st.empty()
            
            frames = []
            frame_count = 0
            
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
                    frames_container.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    if len(frames) >= 5:
                        prompt = """
                        Analyze these video frames for signs of harassment or concerning behavior.
                        Focus on:
                        1. Aggressive or threatening movements
                        2. Signs of distress or danger
                        3. Unsafe situations
                        4. Suspicious patterns
                        
                        Based on training with Bengali content, provide a detailed assessment.
                        """
                        
                        chat = self.model.start_chat(history=[])
                        response = chat.send_message([prompt, *frames])
                        current_time = datetime.now().strftime("%H:%M:%S")
                        analysis_container.markdown(f"""
                        ### 🎥 Analysis Update ({current_time})
                        **Segment:** Frames {frame_count-5} to {frame_count}
                        
                        {response.text}
                        ---
                        """)
                        frames = []
            
            cap.release()
            return "Video analysis completed"
            
        except Exception as e:
            self.logger.error(f"Error in video analysis: {e}")
            return None

    def analyze_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path)
            features = {
                'duration': float(len(y) / sr),
                'rms_energy': float(librosa.feature.rms(y=y).mean()),
                'spectral_centroid': float(librosa.feature.spectral_centroid(y=y).mean()),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean())
            }
            
            prompt = f"""
            Analyze these audio characteristics for concerning content:
            Duration: {features['duration']:.2f} seconds
            Energy Level: {features['rms_energy']:.4f}
            Spectral Features: {features['spectral_centroid']:.2f}
            Voice Patterns: {features['zero_crossing_rate']:.4f}
            
            Based on training with Bengali audio content:
            1. Identify concerning speech patterns
            2. Detect aggressive or threatening tones
            3. Analyze emotional indicators
            4. Note suspicious audio elements
            """
            
            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in audio analysis: {e}")
            return None

    def save_results(self, results):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f"analysis_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return output_file

def main():
    st.title("Content Analysis System")
    
    st.markdown("""
        <style>
        .stVideo {width: 100%; height: auto;}
        .css-1y4p8pa {padding-top: 0rem;}
        .block-container {padding-top: 1rem;}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        st.markdown("---")
        st.markdown("""
        ### Features
        - Real-time video analysis
        - Audio characteristics analysis
        - Automatic result saving
        - Download results
        """)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Video Player")
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
        
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.getbuffer())
                st.video(tmp_video.name)
                with st.expander("File Details"):
                    st.json({
                        "FileName": video_file.name,
                        "FileType": video_file.type,
                        "FileSize": f"{video_file.size / 1024:.2f} KB"
                    })
                try:
                    os.unlink(tmp_video.name)
                except:
                    pass

        if audio_file is not None:
            st.subheader("Audio Player")
            st.audio(audio_file)
            with st.expander("File Details"):
                st.json({
                    "FileName": audio_file.name,
                    "FileType": audio_file.type,
                    "FileSize": f"{audio_file.size / 1024:.2f} KB"
                })

    with col2:
        st.subheader("Real-time Analysis")
        analyze_button = st.button(
            "Start Analysis",
            disabled=not api_key or (not video_file and not audio_file),
            type="primary"
        )

        if analyze_button:
            if api_key:
                analyzer = ContentAnalyzer(api_key=api_key)
                results = analyzer.analyze_content(video_file, audio_file)
                
                if "error" not in results:
                    st.success("Analysis complete!")
                    st.download_button(
                        "Download Analysis Report (JSON)",
                        data=json.dumps(results, indent=2),
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Analysis failed: {results['error']}")
            else:
                st.warning("Please enter your Gemini API key")

if __name__ == "__main__":
    main()
