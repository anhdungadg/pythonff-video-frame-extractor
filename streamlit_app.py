import streamlit as st
import tempfile
import os
import json
import cv2
import numpy as np
from frame_extractor import extract_frames, analyze_video_frames, get_bedrock_client
from scene_detector import detect_scene_changes, detect_scene_changes_nova_pro, detect_scene_changes_bedrock

# Constants
DEFAULT_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
AVAILABLE_MODELS = [
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0'
]

st.set_page_config(
    page_title="Video Frame Extractor and Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎬 Video Frame Extractor and Analyzer")
    
    # Sidebar for app selection
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Frame Extraction", "Scene Detection"]
    )
    
    # AWS Bedrock model selection
    with st.sidebar.expander("AWS Bedrock Settings"):
        selected_model = st.selectbox(
            "Select Claude model",
            options=AVAILABLE_MODELS,
            index=0
        )
        
        # Custom prompt option
        use_custom_prompt = st.checkbox("Use custom prompt")
        custom_prompt = None
        if use_custom_prompt:
            custom_prompt = st.text_area("Enter custom prompt", height=150)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Create temp file to save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Create temp directory for frames
        output_dir = tempfile.mkdtemp()
        
        if app_mode == "Frame Extraction":
            frame_extraction_ui(video_path, output_dir, selected_model, custom_prompt)
        else:  # Scene Detection
            scene_detection_ui(video_path, output_dir, selected_model)
        
        # Cleanup temp files when session ends
        st.session_state['temp_files'] = [video_path, output_dir]

def frame_extraction_ui(video_path, output_dir, model_id=DEFAULT_MODEL_ID, custom_prompt=None):
    st.header("Frame Extraction")
    
    # Get video info
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    video.release()
    
    st.sidebar.subheader("Video Information")
    st.sidebar.info(f"""
    - Total frames: {total_frames}
    - FPS: {fps:.2f}
    - Duration: {duration:.2f} seconds
    """)
    
    # Percentage slider
    percentage = st.sidebar.slider("Percentage of frames to extract", 0.1, 100.0, 10.0, 0.1)
    
    # Calculate number of frames to extract
    num_frames = int(total_frames * percentage / 100)
    st.sidebar.text(f"Will extract approximately {num_frames} frames")
    
    # AWS Bedrock analysis option
    analyze_with_bedrock = st.sidebar.checkbox("Analyze frames with AWS Bedrock")
    
    if st.sidebar.button("Extract Frames"):
        with st.spinner('Extracting frames...'):
            extract_frames(video_path, output_dir, percentage)
            
            # Get list of extracted frames
            frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
            frames.sort()
            
            if len(frames) > 0:
                st.success(f"Successfully extracted {len(frames)} frames!")
                
                # Display frames in a grid
                cols = 3
                rows = (len(frames) + cols - 1) // cols
                
                for i in range(min(rows, 5)):  # Limit to 5 rows initially
                    row_frames = frames[i*cols:min((i+1)*cols, len(frames))]
                    columns = st.columns(len(row_frames))
                    
                    for j, frame in enumerate(row_frames):
                        frame_path = os.path.join(output_dir, frame)
                        columns[j].image(frame_path, caption=frame, use_column_width=True)
                
                if rows > 5:
                    with st.expander("Show more frames"):
                        for i in range(5, rows):
                            row_frames = frames[i*cols:min((i+1)*cols, len(frames))]
                            if not row_frames:
                                break
                                
                            columns = st.columns(len(row_frames))
                            for j, frame in enumerate(row_frames):
                                frame_path = os.path.join(output_dir, frame)
                                columns[j].image(frame_path, caption=frame, use_column_width=True)
                
                # Analyze with AWS Bedrock if selected
                if analyze_with_bedrock:
                    with st.spinner('Analyzing frames with AWS Bedrock...'):
                        try:
                            # Test Bedrock connection first
                            bedrock = get_bedrock_client()
                            if not bedrock:
                                st.error("Could not connect to AWS Bedrock. Check your credentials and region settings.")
                                return
                                
                            results = analyze_video_frames(output_dir, model_id, custom_prompt)
                            
                            if not results:
                                st.error("No analysis results were returned.")
                                return
                                
                            # Check if any result contains an error
                            has_errors = any("Error:" in result.get('analysis', '') for result in results)
                            if has_errors:
                                st.warning("Some frames could not be analyzed. Check the results for details.")
                            
                            # Display analysis results
                            st.subheader("Analysis Results")
                            for result in results:
                                with st.expander(f"Frame: {result['frame']}"):
                                    # Load and display the frame
                                    frame_path = os.path.join(output_dir, result['frame'])
                                    st.image(frame_path)
                                    st.write(result['analysis'])
                            
                            # Save results button
                            results_json = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="Download Analysis Results",
                                data=results_json,
                                file_name="analysis_results.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error analyzing frames: {str(e)}")
                            st.info("Make sure you have AWS credentials configured with access to Bedrock services.")
            else:
                st.warning("No frames were extracted. Try increasing the percentage.")

def scene_detection_ui(video_path, output_dir, model_id=DEFAULT_MODEL_ID):
    st.header("Scene Detection")
    
    # Get video info
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    video.release()
    
    st.sidebar.subheader("Video Information")
    st.sidebar.info(f"""
    - Total frames: {total_frames}
    - FPS: {fps:.2f}
    - Duration: {duration:.2f} seconds
    """)
    
    # Detection method
    detection_method = st.sidebar.selectbox(
        "Detection method",
        ["basic", "nova_pro", "bedrock"],
        format_func=lambda x: {
            "basic": "Basic (Faster)",
            "nova_pro": "Nova Pro (More accurate)",
            "bedrock": "AWS Bedrock (Cloud-based)"
        }.get(x, x)
    )
    
    # Method-specific parameters
    if detection_method == "basic":
        threshold = st.sidebar.slider("Threshold (0-255)", 10, 100, 30, 1)
    else:
        threshold = st.sidebar.slider("Threshold (0-1)", 0.1, 0.9, 0.35, 0.05)
    
    min_scene_length = st.sidebar.slider("Minimum scene length (frames)", 5, 60, 15, 1)
    
    if detection_method == "bedrock":
        sample_rate = st.sidebar.slider("Sample rate (process every nth frame)", 1, 30, 5, 1)
    
    if st.sidebar.button("Detect Scenes"):
        with st.spinner(f'Detecting scenes using {detection_method} method...'):
            try:
                if detection_method == "basic":
                    detect_scene_changes(video_path, output_dir, threshold, min_scene_length)
                elif detection_method == "nova_pro":
                    detect_scene_changes_nova_pro(video_path, output_dir, threshold, min_scene_length)
                else:  # bedrock
                    # Test Bedrock connection first
                    bedrock = get_bedrock_client()
                    if not bedrock:
                        st.error("Could not connect to AWS Bedrock. Check your credentials and region settings.")
                        return
                        
                    detect_scene_changes_bedrock(video_path, output_dir, threshold, min_scene_length, sample_rate, model_id)
                
                # Get list of detected scenes
                scenes = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
                scenes.sort()
                
                if len(scenes) > 0:
                    st.success(f"Successfully detected {len(scenes)} scene changes!")
                    
                    # Display scenes in a grid
                    cols = 2
                    rows = (len(scenes) + cols - 1) // cols
                    
                    for i in range(rows):
                        row_scenes = scenes[i*cols:min((i+1)*cols, len(scenes))]
                        if not row_scenes:
                            break
                            
                        columns = st.columns(len(row_scenes))
                        for j, scene in enumerate(row_scenes):
                            scene_path = os.path.join(output_dir, scene)
                            columns[j].image(scene_path, caption=scene, use_column_width=True)
                            
                            # If there's a description file for this scene (from bedrock method)
                            desc_path = os.path.join(output_dir, f"{scene.split('.')[0]}_desc.txt")
                            if os.path.exists(desc_path):
                                with open(desc_path, 'r', encoding='utf-8') as f:
                                    description = f.read()
                                columns[j].write(description)
                else:
                    st.warning("No scene changes were detected. Try adjusting the threshold.")
            except Exception as e:
                st.error(f"Error detecting scenes: {str(e)}")
                if detection_method == "nova_pro":
                    st.info("Make sure you have PyTorch and Transformers installed.")
                elif detection_method == "bedrock":
                    st.info("Make sure you have AWS credentials configured with access to Bedrock services.")

if __name__ == "__main__":
    # Initialize session state for temp file cleanup
    if 'temp_files' not in st.session_state:
        st.session_state['temp_files'] = []
    
    main()
    
    # Cleanup temp files when app is rerun
    for temp_file in st.session_state.get('temp_files', []):
        if os.path.isfile(temp_file):
            os.unlink(temp_file)
        elif os.path.isdir(temp_file):
            for file in os.listdir(temp_file):
                os.unlink(os.path.join(temp_file, file))
            os.rmdir(temp_file)