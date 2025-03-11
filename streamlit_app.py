import streamlit as st
import os
import json
import tempfile
from datetime import datetime
import logging
import numpy as np
from PIL import Image
import cv2

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

# Configure logging - direct to streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock AI model for testing (remove AI dependency for debugging)
class MockAIModel:
    def analyze_frames(self, frames):
        return """
        Safety Assessment:
        - Type of concern: Potential intimidation scenario detected
        - Severity: Medium
        - Recommendation: Increased awareness advised in this environment
        
        The video shows a situation that could potentially escalate, with signs of 
        uncomfortable body language and positioning that may indicate unwanted attention
        or intimidation. Monitoring is recommended.
        """
    
    def analyze_audio(self, duration, energy):
        return """
        Audio Analysis:
        - Type of concern: Elevated vocal stress detected
        - Urgency: Medium
        - Recommendation: Follow-up monitoring recommended
        
        The audio contains patterns consistent with distress or discomfort, 
        though not immediate danger. The energy levels and patterns suggest 
        an uncomfortable situation that should be monitored.
        """

class WomenSafetyAnalyzer:
    def __init__(self):
        # Use mock AI for testing
        self.model = MockAIModel()
        
    def analyze_video_safety(self, video_path):
        try:
            st.info("Starting video analysis...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error(f"Failed to open video file: {video_path}")
                return ["Failed to open video file"]
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                st.warning("Video appears to be empty or unreadable")
                return ["Video appears to be empty or unreadable"]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frames = []
            frame_count = 0
            safety_concerns = []
            
            # Debug info
            st.write(f"Video info: {fps} FPS, {total_frames} frames")
            
            # Process less frames for testing
            sample_rate = max(int(fps), 1) * 5  # One frame every 5 seconds
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress more frequently
                if frame_count % 10 == 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing frame {frame_count}/{total_frames}")
                
                if frame_count % sample_rate == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                    
                    # Analyze after collecting enough frames or at the end
                    if len(frames) >= 3 or frame_count >= total_frames-10:
                        status_text.text(f"Running AI analysis on collected frames...")
                        # Use mock AI for predictable testing
                        analysis = self.model.analyze_frames(frames)
                        safety_concerns.append(analysis)
                        frames = []
            
            cap.release()
            progress_bar.progress(100)
            status_text.text("Video analysis complete")
            
            # Ensure we return something even if no frames were processed
            if not safety_concerns:
                safety_concerns = ["No significant safety concerns detected in the video."]
                
            return safety_concerns
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            st.error(f"Video analysis error: {str(e)}")
            return [f"Error: {str(e)}"]

    def analyze_audio_safety(self, audio_path):
        try:
            st.info("Starting audio analysis...")
            
            # Mock audio analysis to avoid dependencies
            duration = 30.0  # seconds
            rms_energy = 0.18  # mock value
            
            # Use mock AI for testing
            analysis = self.model.analyze_audio(duration, rms_energy)
            return analysis
        
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            st.error(f"Audio analysis error: {str(e)}")
            return f"Error: {str(e)}"

    def analyze_content(self, video_file, audio_file):
        results = {
            "timestamp": datetime.now().isoformat(),
            "video_analysis": None,
            "audio_analysis": None
        }

        if video_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(video_file.getvalue())
                    tmp_path = tmp_video.name
                
                st.write(f"Saved video to temporary file: {tmp_path}")
                results["video_analysis"] = self.analyze_video_safety(tmp_path)
                
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                results["video_analysis"] = [f"Error: {str(e)}"]

        if audio_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(audio_file.getvalue())
                    tmp_path = tmp_audio.name
                
                results["audio_analysis"] = self.analyze_audio_safety(tmp_path)
                
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                results["audio_analysis"] = f"Error: {str(e)}"

        return results

def main():
    # Display header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #FF69B4, #FFB6C1); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h1 style='color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>Women Safety Analysis System</h1>
            <p style='color: white; font-size: 1.2rem;'>Empowering Safety Through Technology</p>
        </div>
    """, unsafe_allow_html=True)

    # Debug info - show this first
    st.write("Application started. Debug mode enabled.")

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
        
        # Add dependency information
        st.markdown("---")
        st.markdown("""
        ### System Information
        This is a test version with mock AI responses.
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
            st.write("Video preview:")
            st.video(video_file)
        if audio_file:
            st.write("Audio preview:")
            st.audio(audio_file)

    with col2:
        st.subheader("Analysis Results")
        
        # Add debug button
        if st.button("Test Connection", key="test_connection"):
            st.success("Application is responding correctly!")
        
        analyze_button = st.button("Begin Safety Analysis", 
                                type="primary",
                                use_container_width=True,
                                disabled=(not video_file and not audio_file))

        if analyze_button:
            st.write("Analysis button clicked")
            analyzer = WomenSafetyAnalyzer()
            
            with st.spinner("Analyzing content for safety concerns..."):
                results = analyzer.analyze_content(video_file, audio_file)
                
                # Always show results structure for debugging
                st.write("Results structure:", type(results))
                
                if results.get("video_analysis") or results.get("audio_analysis"):
                    st.markdown("""
                        <div style='background: linear-gradient(45deg, #2b1f47, #1a1a2e); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <h3 style='color: #FF69B4; margin-bottom: 1rem;'>Safety Assessment</h3>
                    """, unsafe_allow_html=True)
                    
                    if results.get("video_analysis"):
                        st.markdown("<h4 style='color: white;'>Video Analysis:</h4>", unsafe_allow_html=True)
                        for i, analysis in enumerate(results["video_analysis"]):
                            st.markdown(f"""
                                <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                                    <div style='color: white;'>{analysis}</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if results.get("audio_analysis"):
                        st.markdown("<h4 style='color: white;'>Audio Analysis:</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;'>
                                <div style='color: white;'>{results["audio_analysis"]}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Download button for report
                    st.download_button(
                        label="Download Safety Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_report"
                    )
                else:
                    st.error("Analysis failed or no safety concerns detected.")
                    
        # Add information about system usage
        with st.expander("How to use this system"):
            st.markdown("""
            1. **Upload Media**: Add video and/or audio files for analysis
            2. **Begin Analysis**: Click the analysis button to start the AI-powered safety assessment
            3. **Review Results**: The system will highlight potential safety concerns and their severity
            4. **Download Report**: Save the analysis for documentation or further action
            
            This system is designed to assist in identifying potential safety concerns for women in various environments.
            It is not a replacement for emergency services - if you suspect immediate danger, contact local authorities.
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)
