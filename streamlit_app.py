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
    page_title="Women Safety Analysis System",
    page_icon="🚺",
    layout="wide"
)

class WomenSafetyAnalyzer:
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
        self.output_dir = "safety_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_content(self, video_file, audio_file):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "video_safety_analysis": None,
                "audio_safety_analysis": None,
                "risk_level": None,
                "safety_recommendations": None
            }

            if video_file is not None:
                self.logger.info("Analyzing video for safety concerns...")
                progress_bar.progress(25)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(video_file.read())
                    video_analysis = self.analyze_video_safety(tmp_video.name)
                    results["video_safety_analysis"] = video_analysis
                os.unlink(tmp_video.name)
                progress_bar.progress(75)

            if audio_file is not None:
                self.logger.info("Analyzing audio for distress signals...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    audio_analysis = self.analyze_audio_safety(tmp_audio.name)
                    results["audio_safety_analysis"] = audio_analysis
                os.unlink(tmp_audio.name)

            # Determine overall risk level
            if results["video_safety_analysis"] or results["audio_safety_analysis"]:
                results["risk_level"] = self.assess_risk_level(results)
                results["safety_recommendations"] = self.generate_safety_recommendations(results["risk_level"])

            self.save_results(results)
            return results

        except Exception as e:
            self.logger.error(f"Error in safety analysis: {e}")
            return {"error": str(e)}

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
                if frame_count % int(fps) == 0:  # Analyze one frame per second
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    frames.append(pil_frame)
                    
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(progress)
                    frames_container.text(f"Analyzing frame {frame_count}/{total_frames} for safety concerns")
                    
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

                        Provide a detailed safety assessment with:
                        - Specific safety concerns identified
                        - Risk level (Low/Medium/High)
                        - Timestamp of concerning behavior
                        - Description of the concerning action
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
            
            safety_audio_prompt = f"""
            Analyze these audio characteristics specifically for women's safety concerns:
            Duration: {features['duration']:.2f} seconds
            Energy Level: {features['rms_energy']:.4f} (indicating volume/intensity)
            Spectral Features: {features['spectral_centroid']:.2f} (indicating voice pitch)
            Voice Patterns: {features['zero_crossing_rate']:.4f} (indicating speech patterns)
            
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

            3. Emotional Indicators:
               - Fear or panic in voice
               - Signs of verbal intimidation
               - Defensive or distressed responses
               - Escalating confrontational patterns

            Provide a detailed safety assessment with:
            - Specific audio safety concerns
            - Risk level assessment
            - Timestamps of concerning sounds
            - Recommended actions
            """
            
            chat = self.model.start_chat(history=[])
            response = chat.send_message(safety_audio_prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in audio safety analysis: {e}")
            return None

    def assess_risk_level(self, results):
        # Analyze results to determine overall risk level
        risk_levels = {
            "HIGH": ["assault", "violence", "forced", "threatening", "danger", "emergency"],
            "MEDIUM": ["harassment", "following", "uncomfortable", "suspicious", "intimidation"],
            "LOW": ["uncertain", "possible", "mild", "potential", "unclear"]
        }
        
        combined_analysis = str(results.get("video_safety_analysis", "")) + str(results.get("audio_safety_analysis", ""))
        combined_analysis = combined_analysis.lower()
        
        for level, keywords in risk_levels.items():
            if any(keyword in combined_analysis for keyword in keywords):
                return level
                
        return "LOW"

    def generate_safety_recommendations(self, risk_level):
        recommendations = {
            "HIGH": [
                "Immediately contact emergency services (911)",
                "Alert nearby security personnel or authorities",
                "Document the incident with timestamps and details",
                "Seek immediate assistance from nearby people",
                "Move to a well-lit, populated area if possible"
            ],
            "MEDIUM": [
                "Stay alert and aware of surroundings",
                "Contact a trusted friend or family member",
                "Move to a more populated area",
                "Consider recording the situation discretely",
                "Be prepared to contact authorities if situation escalates"
            ],
            "LOW": [
                "Stay aware of surroundings",
                "Trust your instincts if something feels wrong",
                "Keep emergency contacts readily available",
                "Consider walking with a friend or group",
                "Note any changes in the situation"
            ]
        }
        return recommendations.get(risk_level, [])

    def save_results(self, results):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f"safety_analysis_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return output_file

def main():
    # Hardcoded API key
    api_key = "AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E"
    
    st.markdown("""
        <div class="header-container">
            <div class="flower flower-1"></div>
            <div class="flower flower-2"></div>
            <div class="flower flower-3"></div>
            <div class="flower flower-4"></div>
            <h1 class="header-title">Women Safety Analysis System</h1>
            <p class="header-subtitle">Empowering Safety Through Technology</p>
        </div>
    """, unsafe_allow_html=True)
    
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

        .browse-button {
            background: rgba(255, 105, 180, 0.2);
            color: #FF69B4;
            padding: 0.4rem 0.8rem;
            border-radius: 5px;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .browse-button:hover {
            background: rgba(255, 105, 180, 0.3);
        }

        /* Flower Decorations */
        .flower {
            position: absolute;
            width: 30px;
            height: 30px;
            background: white;
            border-radius: 50%;
        }
        
        .flower::before {
            content: '🌸';
            font-size: 24px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        
        .flower-1 { top: 20px; left: 20px; }
        .flower-2 { top: 20px; right: 20px; }
        .flower-3 { bottom: 20px; left: 20px; }
        .flower-4 { bottom: 20px; right: 20px; }
        </style>
    """, unsafe_allow_html=True)

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
        
        video_file = st.file_uploader("Upload Video for Safety Analysis", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
        audio_file = st.file_uploader("Upload Audio for Distress Analysis", type=['wav', 'mp3'], label_visibility="collapsed")
        
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.getbuffer())
                st.video(tmp_video.name)
                with st.expander("File Information"):
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
            st.subheader("Audio Analysis")
            st.audio(audio_file)
            with st.expander("File Information"):
                st.json({
                    "FileName": audio_file.name,
                    "FileType": audio_file.type,
                    "FileSize": f"{audio_file.size / 1024:.2f} KB"
                })

    with col2:
        st.subheader("Safety Analysis Results")
        analyze_button = st.button(
            "Begin Safety Analysis",
            disabled=(not video_file and not audio_file),
            type="primary"
        )

        if analyze_button:
            analyzer = WomenSafetyAnalyzer(api_key=api_key)
            results = analyzer.analyze_content(video_file, audio_file)
            
            if "error" not in results:
                risk_level = results.get("risk_level", "LOW")
                st.markdown(f"""
                ### Overall Risk Assessment
                <div class='{risk_level.lower()}-risk'>Risk Level: {risk_level}</div>
                """, unsafe_allow_html=True)
                
                st.subheader("Safety Recommendations")
                for rec in results.get("safety_recommendations", []):
                    st.warning(rec)
                
                st.success("Safety analysis complete!")
                st.download_button(
                    "Download Safety Report (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.error(f"Safety analysis failed: {results['error']}")
            else:
                st.warning("Please enter your Gemini API key to begin analysis")

if __name__ == "__main__":
    main()
