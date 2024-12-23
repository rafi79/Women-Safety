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
    .high-risk { color: #ff0000; font-weight: bold; }
    .medium-risk { color: #ffa500; font-weight: bold; }
    .low-risk { color: #008000; font-weight: bold; }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #FF69B4, #FFB6C1);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        font-family: 'Arial', sans-serif;
    }
    
    .header-subtitle {
        color: white;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-family: 'Arial', sans-serif;
    }
    
    /* Results Section Styling */
    .results-container {
        background: linear-gradient(45deg, #2b1f47, #1a1a2e);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .risk-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .risk-high {
        border-color: #ff4444;
        background: rgba(255, 68, 68, 0.1);
    }
    
    .risk-medium {
        border-color: #ffbb33;
        background: rgba(255, 187, 51, 0.1);
    }
    
    .risk-low {
        border-color: #00C851;
        background: rgba(0, 200, 81, 0.1);
    }
    
    .risk-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .risk-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    
    .risk-title {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .analysis-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    /* Upload Box Styling */
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .upload-title {
        color: #FF69B4;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .upload-box {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        min-height: 60px;
    }
    
    .upload-icon {
        color: #FF69B4;
        font-size: 1.5rem;
    }
    
    .upload-limit {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

class WomenSafetyAnalyzer:
    def __init__(self):
        # Initialize Gemini API
        genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = "safety_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_video_safety(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            frames_container = st.empty()
            analysis_container = st.empty()
            
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
                    frames_container.text(f"Analyzing frame {frame_count}/{total_frames}")
                    
                    if len(frames) >= 5:
                        safety_prompt = """
                        Analyze these video frames specifically for women's safety concerns. Focus on detecting:

                        1. Physical Safety Issues:
                           - Unwanted physical contact or proximity
                           - Aggressive or threatening body language
                           - Physical harassment or assault attempts
                           - Forceful restraint or blocking movements

                        2. Signs of Distress:
                           - Defensive body language
                           - Signs of fear or discomfort
                           - Attempts to create distance
                           - Facial expressions indicating distress

                        3. Environmental Safety:
                           - Suspicious following or stalking behavior
                           - Unwanted photography or recording
                           - Corner or isolation tactics
                           - Group intimidation scenarios

                        4. Harassment Indicators:
                           - Inappropriate touching or gestures
                           - Invasion of personal space
                           - Unwanted advances
                           - Threatening postures
                        """
                        
                        chat = self.model.start_chat(history=[])
                        response = chat.send_message([safety_prompt, *frames])
                        current_time = datetime.now().strftime("%H:%M:%S")
                        
                        analysis_text = response.text
                        safety_concerns.append({
                            "timestamp": current_time,
                            "frame_range": f"{frame_count-5} to {frame_count}",
                            "analysis": analysis_text
                        })
                        
                        analysis_container.markdown(f"""
                        ### 🚨 Safety Analysis Update ({current_time})
                        **Segment:** Frames {frame_count-5} to {frame_count}
                        
                        {analysis_text}
                        ---
                        """)
                        frames = []
            
            cap.release()
            return safety_concerns
            
        except Exception as e:
            self.logger.error(f"Error in video safety analysis: {e}")
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
            Spectral Features: {features['spectral_centroid']:.2f}
            Voice Patterns: {features['zero_crossing_rate']:.4f}
            
            Focus on detecting:
            1. Verbal Safety Concerns:
               - Distressed vocals or crying
               - Verbal harassment or threats
               - Calls for help or assistance
               - Aggressive or threatening tones

            2. Environmental Audio:
               - Background sounds indicating unsafe situations
               - Multiple voices indicating group harassment
               - Sounds of pursuit or running
               - Indicators of physical struggle
            """
            
            chat = self.model.start_chat(history=[])
            response = chat.send_message(audio_prompt)
            return response.text
        
        except Exception as e:
            self.logger.error(f"Error in audio safety analysis: {e}")
            return None

    def calculate_risk_factors(self, analysis_text):
        risk_factors = {
            'high': {
                'keywords': ['assault', 'violence', 'forced', 'threatening', 'danger', 'emergency'],
                'found': []
            },
            'medium': {
                'keywords': ['harassment', 'following', 'uncomfortable', 'suspicious', 'intimidation'],
                'found': []
            },
            'low': {
                'keywords': ['uncertain', 'possible', 'mild', 'potential', 'unclear'],
                'found': []
            }
        }
        
        text = analysis_text.lower()
        for level in risk_factors:
            for keyword in risk_factors[level]['keywords']:
                if keyword in text:
                    risk_factors[level]['found'].append(keyword)
        
        return risk_factors

    def analyze_content(self, video_file, audio_file):
        results = {
            "timestamp": datetime.now().isoformat(),
            "video_analysis": None,
            "audio_analysis": None,
            "risk_factors": {
                "high": [],
                "medium": [],
                "low": []
            },
            "overall_risk_level": "LOW"
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
        <div class="header-container">
            <h1 class="header-title">Women Safety Analysis System</h1>
            <p class="header-subtitle">Empowering Safety Through Technology</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.markdown("""
        ### Safety Analysis Features
        - Real-time safety threat detection
        - Distress signal analysis
        - Risk level assessment
        - Safety recommendations
        - Automatic incident reporting
        - Emergency response guidance
        """)

    # Main content layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="upload-container">
                <div class="upload-title">📹 Video Analysis</div>
                <div class="upload-box">
                    <span class="upload-icon">⬆️</span>
                    <div style="flex-grow: 1;">
                        <div>Drag and drop video here</div>
                        <div class="upload-limit">Limit 200MB per file • MP4, AVI, MOV, MPEG4</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")

        st.markdown("""
            <div class="upload-container">
                <div class="upload-title">🎤 Audio Analysis</div>
                <div class="upload-box">
                    <span class="upload-icon">⬆️</span>
                    <div style="flex-grow: 1;">
                        <div>Drag and drop audio here</div>
                        <div class="upload-limit">Limit 200MB per file • WAV, MP3</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3'], label_visibility="collapsed")

        if video_file:
            st.video(video_file)

        if audio_file:
            st.audio(audio_file)

    with col2:
        st.subheader("Safety Analysis Results")
        analyze_button = st.button("Begin Safety Analysis", disabled=(not video_file and not audio_file), type="primary")

        if analyze_button:
            analyzer = WomenSafetyAnalyzer()
            with st.spinner("Analyzing content for safety concerns..."):
                results = analyzer.analyze_content(video_file, audio_file)
                
                if results.get("video_analysis") or results.get("audio_analysis"):
                    st.success("Analysis complete!")
                    
                    # Single unified analysis output
                    st.markdown("""
                        <div style='background: linear-gradient(45deg, #2b1f47, #1a1a2e); border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; margin-bottom: 1.5rem;'>Safety Analysis Report</h2>
                            
                            <div style='background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                                <div style='color: #FF69B4; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem;'>
                                    Critical Safety Indicators:
                                </div>
                                <div style='color: white; padding-left: 20px;'>
                                    • Unwanted Physical Contact Detection<br/>
                                    • Harassment Pattern Analysis<br/>
                                    • Distress Signal Monitoring<br/>
                                    • Aggressive Behavior Recognition<br/>
                                    • Unsafe Environment Detection<br/>
                                    • Personal Space Violation Alerts
                                </div>
                            </div>

                            <div style='margin-top: 1.5rem;'>
                                <div style='color: #FF69B4; font-weight: bold; margin-bottom: 0.5rem;'>Analysis Summary</div>
                                <div style='color: white; background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;'>
                    """, unsafe_allow_html=True)
                    
                    # Combine and summarize analysis results
                    if results.get("video_analysis"):
                        analysis = results["video_analysis"][0]["analysis"] if results["video_analysis"] else ""
                        st.markdown(f"<div style='color: white;'>{analysis}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div></div></div>", unsafe_allow_html=True)

                    # Download button for detailed report
                    st.download_button(
                        "Download Complete Analysis Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Offer download of complete analysis
                    st.download_button(
                        "Download Complete Analysis Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("No safety concerns detected or analysis failed.")

if __name__ == "__main__":
    main()
