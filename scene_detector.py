import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
import boto3
import json
import base64
from io import BytesIO
from PIL import Image

# Constants
DEFAULT_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
DEFAULT_REGION = 'us-east-1'

def detect_scene_changes(video_path, output_folder, threshold=30, min_scene_length=15):
    """
    Detects scene changes in a video and extracts frames after cuts
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames
        threshold (int): Threshold for scene change detection (0-255)
        min_scene_length (int): Minimum number of frames between scene changes
        
    Returns:
        list: List of frame indices where scene changes occur
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return []
    
    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Read the first frame
    success, prev_frame = video.read()
    if not success:
        print("Could not read the first frame")
        return []
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 1
    scene_count = 0
    frames_since_last_scene = 0
    
    # List to store frame indices where scene changes occur
    scene_changes = []
    
    # Process all frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        # Read the next frame
        success, curr_frame = video.read()
        
        if not success:
            break
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        
        # Calculate mean difference
        mean_diff = np.mean(frame_diff)
        
        # Detect scene change if mean difference is above threshold and minimum scene length is satisfied
        if mean_diff > threshold and frames_since_last_scene >= min_scene_length:
            scene_changes.append(frame_count)
            
            # Save the frame after the cut
            output_path = os.path.join(output_folder, f"scene_{scene_count:04d}.jpg")
            cv2.imwrite(output_path, curr_frame)
            
            scene_count += 1
            frames_since_last_scene = 0
        else:
            frames_since_last_scene += 1
        
        # Update previous frame
        prev_gray = curr_gray
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    video.release()
    
    print(f"Completed! Detected {scene_count} scene changes and saved frames to {output_folder}")
    return scene_changes


def detect_scene_changes_nova_pro(video_path, output_folder, threshold=0.35, min_scene_length=15):
    """
    Detects scene changes in a video using Nova Pro model and extracts frames after cuts
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames
        threshold (float): Threshold for scene change detection (0-1)
        min_scene_length (int): Minimum number of frames between scene changes
        
    Returns:
        list: List of frame indices where scene changes occur
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        print("Required packages not found. Please install with:")
        print("pip install torch transformers")
        return []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load Nova Pro model
    try:
        print("Loading Nova Pro model...")
        processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading Nova Pro model: {e}")
        return []
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return []
    
    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Read the first frame
    success, prev_frame = video.read()
    if not success:
        print("Could not read the first frame")
        return []
    
    # Convert BGR to RGB for the model
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    
    # Get embedding for the first frame
    with torch.no_grad():
        inputs = processor(images=prev_frame_rgb, return_tensors="pt").to(device)
        prev_embedding = model(**inputs).last_hidden_state.mean(dim=1)
    
    frame_count = 1
    scene_count = 0
    frames_since_last_scene = 0
    
    # List to store frame indices where scene changes occur
    scene_changes = []
    
    # Process all frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        # Read the next frame
        success, curr_frame = video.read()
        
        if not success:
            break
        
        # Process every 3rd frame to speed up (adjust as needed)
        if frame_count % 3 != 0 and frame_count > 1:
            frame_count += 1
            pbar.update(1)
            frames_since_last_scene += 1
            continue
        
        # Convert BGR to RGB for the model
        curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        
        # Get embedding for current frame
        with torch.no_grad():
            inputs = processor(images=curr_frame_rgb, return_tensors="pt").to(device)
            curr_embedding = model(**inputs).last_hidden_state.mean(dim=1)
        
        # Calculate cosine similarity between embeddings
        similarity = torch.nn.functional.cosine_similarity(prev_embedding, curr_embedding).item()
        difference = 1 - similarity
        
        # Detect scene change if difference is above threshold and minimum scene length is satisfied
        if difference > threshold and frames_since_last_scene >= min_scene_length:
            scene_changes.append(frame_count)
            
            # Save the frame after the cut
            output_path = os.path.join(output_folder, f"scene_{scene_count:04d}.jpg")
            cv2.imwrite(output_path, curr_frame)
            
            scene_count += 1
            frames_since_last_scene = 0
        else:
            frames_since_last_scene += 1
        
        # Update previous embedding
        prev_embedding = curr_embedding
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    video.release()
    
    print(f"Completed! Detected {scene_count} scene changes and saved frames to {output_folder}")
    return scene_changes


def get_bedrock_client(region=DEFAULT_REGION):
    """
    Initialize and return AWS Bedrock client
    
    Args:
        region (str): AWS region name
        
    Returns:
        boto3.client: Initialized Bedrock client or None if failed
    """
    try:
        return boto3.client(service_name='bedrock-runtime', region_name=region)
    except Exception as e:
        print(f"Error initializing AWS Bedrock client: {str(e)}")
        print("Make sure you have AWS credentials configured with access to Bedrock services.")
        return None


def get_image_description(image, bedrock_client, model_id=DEFAULT_MODEL_ID):
    """
    Get description of an image using AWS Bedrock with Claude model
    
    Args:
        image (PIL.Image): Image to describe
        bedrock_client: AWS Bedrock client
        model_id (str): Bedrock model ID to use
        
    Returns:
        str: Description of the image
    """
    if not bedrock_client:
        return "Error: No Bedrock client available"
    
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare request body
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_str
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this video frame in a concise way, focusing on the main subjects, setting, and action."
                        }
                    ]
                }
            ]
        })

        # Call Bedrock API
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        description = response_body['content'][0]['text']
        return description
        
    except Exception as e:
        error_message = str(e)
        print(f"Error getting image description: {error_message}")
        
        # Check for specific model-related errors and provide helpful message
        if "ValidationException" in error_message and "model ID" in error_message:
            available_models = ["anthropic.claude-3-sonnet-20240229-v1:0", 
                               "anthropic.claude-3-haiku-20240307-v1:0"]
            return f"Error: The specified model is not available. Try one of these models instead: {', '.join(available_models)}"
        
        return f"Error: {error_message}"


def calculate_description_difference(desc1, desc2):
    """
    Calculate semantic difference between two descriptions
    Simple implementation using word overlap
    
    Args:
        desc1 (str): First description
        desc2 (str): Second description
        
    Returns:
        float: Difference score (0-1)
    """
    if not desc1 or not desc2:
        return 0.5  # Default value if descriptions are missing
    
    # Check if either description contains an error message
    if desc1.startswith("Error:") or desc2.startswith("Error:"):
        return 0.0  # Don't detect scene change on error
    
    # Convert to lowercase and split into words
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    
    # Calculate Jaccard distance
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    similarity = intersection / union
    return 1 - similarity


def detect_scene_changes_bedrock(video_path, output_folder, threshold=0.35, min_scene_length=15, sample_rate=5, model_id=DEFAULT_MODEL_ID):
    """
    Detects scene changes in a video using AWS Bedrock with Claude model
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames
        threshold (float): Threshold for scene change detection (0-1)
        min_scene_length (int): Minimum number of frames between scene changes
        sample_rate (int): Process every nth frame to reduce API calls
        model_id (str): Bedrock model ID to use
        
    Returns:
        list: List of frame indices where scene changes occur
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize Bedrock client
    bedrock = get_bedrock_client()
    if not bedrock:
        return []
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return []
    
    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Read the first frame
    success, prev_frame = video.read()
    if not success:
        print("Could not read the first frame")
        return []
    
    # Convert to PIL Image and get description
    prev_pil_image = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
    prev_description = get_image_description(prev_pil_image, bedrock, model_id)
    
    frame_count = 1
    scene_count = 0
    frames_since_last_scene = 0
    
    # List to store frame indices where scene changes occur
    scene_changes = []
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        # Read the next frame
        success, curr_frame = video.read()
        
        if not success:
            break
        
        # Process every nth frame to reduce API calls
        if frame_count % sample_rate != 0 and frame_count > 1:
            frame_count += 1
            pbar.update(1)
            frames_since_last_scene += 1
            continue
        
        # Convert to PIL Image and get description
        curr_pil_image = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
        curr_description = get_image_description(curr_pil_image, bedrock, model_id)
        
        # Calculate semantic difference between descriptions
        difference = calculate_description_difference(prev_description, curr_description)
        
        # Detect scene change if difference is above threshold and minimum scene length is satisfied
        if difference > threshold and frames_since_last_scene >= min_scene_length:
            scene_changes.append(frame_count)
            
            # Save the frame after the cut
            output_path = os.path.join(output_folder, f"scene_{scene_count:04d}.jpg")
            cv2.imwrite(output_path, curr_frame)
            
            # Save description
            desc_path = os.path.join(output_folder, f"scene_{scene_count:04d}_desc.txt")
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(curr_description)
            
            scene_count += 1
            frames_since_last_scene = 0
        else:
            frames_since_last_scene += 1
        
        # Update previous description
        prev_description = curr_description
        frame_count += 1
        pbar.update(sample_rate)
    
    pbar.close()
    video.release()
    
    print(f"Completed! Detected {scene_count} scene changes and saved frames to {output_folder}")
    return scene_changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scene changes in video and extract frames after cuts")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_folder", help="Folder to save extracted frames")
    parser.add_argument("--method", choices=["basic", "nova_pro", "bedrock"], default="basic", 
                        help="Detection method: basic (faster), nova_pro (more accurate), or bedrock (using AWS)")
    parser.add_argument("--threshold", type=float, default=None, 
                        help="Threshold for scene change detection (0-255 for basic, 0-1 for nova_pro and bedrock)")
    parser.add_argument("--min_scene_length", type=int, default=15,
                        help="Minimum number of frames between scene changes")
    parser.add_argument("--sample_rate", type=int, default=5,
                        help="Process every nth frame (only for bedrock method)")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help="AWS Bedrock model ID (only for bedrock method)")
    
    args = parser.parse_args()
    
    if args.method == "basic":
        threshold = 30 if args.threshold is None else args.threshold
        detect_scene_changes(args.video_path, args.output_folder, threshold, args.min_scene_length)
    elif args.method == "nova_pro":
        threshold = 0.35 if args.threshold is None else args.threshold
        detect_scene_changes_nova_pro(args.video_path, args.output_folder, threshold, args.min_scene_length)
    else:  # bedrock
        threshold = 0.35 if args.threshold is None else args.threshold
        detect_scene_changes_bedrock(args.video_path, args.output_folder, threshold, args.min_scene_length, args.sample_rate, args.model)