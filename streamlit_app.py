import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Content Analysis System",
    page_icon="📊",
    layout="wide"
)

# Rest of the imports
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

class ContentAnalyzer:
    def __init__(self, api_key):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )

        # Setup logging and output directory
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_content(self, video_file, audio_file):
        """Analyze uploaded video and audio content with real-time updates"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "video_analysis": None,
                "audio_analysis": None
            }

            # Analyze video if provided
            if video_file is not None:
                self.logger.info("Analyzing video...")
                status_text.text("Analyzing video frames...")
                progress_bar.progress(25)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(video_file.read())
                    # Create a placeholder for real-time analysis
                    analysis_placeholder = st.empty()
                    
                    # Analyze video with real-time updates
                    video_analysis = self.analyze_video(
                        tmp_video.name, 
                        progress_callback=lambda x: analysis_placeholder.markdown(f"**Current Analysis:**\n{x}")
                    )
                    results["video_analysis"] = video_analysis
                os.unlink(tmp_video.name)
                progress_bar.progress(75)

            # Analyze audio if provided
            if audio_file is not None:
                self.logger.info("Analyzing audio...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    audio_analysis = self.analyze_audio(tmp_audio.name)
                    results["audio_analysis"] = audio_analysis
                os.unlink(tmp_audio.name)

            # Save results
            self.save_results(results)

            return results

        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return {"error": str(e)}

    def analyze_video(self, video_path, progress_callback=None):
        """Analyze video file with real-time frame display and analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create placeholders for frame and analysis
            frame_placeholder = st.empty()
            analysis_placeholder = st.empty()
            
            frames = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % int(fps) == 0:  # Process one frame per second
                    # Convert frame to RGB for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    
                    # Display the current frame
                    frame_placeholder.image(pil_frame, caption=f"Frame {frame_count}")
                    frames.append(pil_frame)
                    
                    # Analyze current batch of frames
                    if len(frames) >= 5:  # Analyze every 5 frames

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
                        
                        # Update analysis display
                        current_analysis = f"""
                        ### Frame {frame_count} Analysis
                        {response.text}
                        """
                        analysis_placeholder.markdown(current_analysis)
                        
                        # Clear frames buffer
                        frames = []
            
            cap.release()
            return "Video analysis complete"

        except Exception as e:
            self.logger.error(f"Error in video analysis: {e}")
            return None

    def analyze_audio(self, audio_path):
        """Analyze audio file"""
        try:
            features = self.extract_audio_features(audio_path)

            prompt = f"""
            Analyze these audio characteristics for concerning content:

            Audio Features:
            - Duration: {features['duration']:.2f} seconds
            - Energy Level: {features['rms_energy']:.4f}
            - Spectral Features: {features['spectral_centroid']:.2f}
            - Voice Patterns: {features['zero_crossing_rate']:.4f}

            Based on training with Bengali audio content:
            1. Identify any concerning speech patterns
            2. Detect aggressive or threatening tones
            3. Analyze emotional indicators
            4. Note any suspicious audio elements
            """

            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            return response.text

        except Exception as e:
            self.logger.error(f"Error in audio analysis: {e}")
            return None

    def extract_frames(self, video_path, max_frames=10):
        """Extract frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max_frames)

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break

        cap.release()
        return frames

    def extract_audio_features(self, audio_path):
        """Extract audio features"""
        y, sr = librosa.load(audio_path)
        return {
            'duration': float(len(y) / sr),
            'rms_energy': float(librosa.feature.rms(y=y).mean()),
            'spectral_centroid': float(librosa.feature.spectral_centroid(y=y).mean()),
            'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean())
        }

    def save_results(self, results):
        """Save analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f"analysis_{timestamp}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to {output_file}")
        return output_file

def main():
    st.title("Content Analysis System")
    st.write("Upload video or audio files for real-time analysis")
    
    # Custom CSS for video player
    st.markdown("""
        <style>
        .stVideo {
            width: 100%;
            height: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'results_file' not in st.session_state:
        st.session_state.results_file = None

    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")

        st.markdown("---")
        st.markdown("""
        ### Features
        - Video content analysis
        - Audio characteristics analysis
        - Automatic result saving
        - Download results
        """)

    # Main content area with two equal columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Video Input")
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            # File info
            file_details = {
                "FileName": video_file.name,
                "FileType": video_file.type,
                "FileSize": f"{video_file.size / 1024:.2f} KB"
            }
            st.write("File Details:", file_details)
            
    with col2:
        st.subheader("Real-time Analysis")
        st.write("Gemini's analysis will appear here as the video plays")

        if st.button("Analyze Content", disabled=not api_key or (not video_file and not audio_file)):
            if api_key:
                with st.spinner("Analyzing content..."):
                    analyzer = ContentAnalyzer(api_key=api_key)
                    st.session_state.results = analyzer.analyze_content(video_file, audio_file)
                    if "error" not in st.session_state.results:
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Analysis failed: {st.session_state.results['error']}")
            else:
                st.warning("Please enter your Gemini API key")

    # Empty middle column for spacing
    with col2:
        st.empty()
        
    # Analysis results column
    with col3:
        st.subheader("Real-time Analysis")
        if st.session_state.results and "error" not in st.session_state.results:
            # Display timestamp
            st.text(f"🕒 Analysis completed at: {st.session_state.results['timestamp']}")

            # Display video analysis
            if st.session_state.results["video_analysis"]:
                st.markdown("### 🎥 Video Analysis")
                st.markdown("---")
                st.write(st.session_state.results["video_analysis"])

            # Display audio analysis
            if st.session_state.results["audio_analysis"]:
                st.markdown("### 🔊 Audio Analysis")
                st.markdown("---")
                st.write(st.session_state.results["audio_analysis"])

            # Download button
            st.download_button(
                label="Download Results (JSON)",
                data=json.dumps(st.session_state.results, indent=2),
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
