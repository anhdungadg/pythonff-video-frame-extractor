import cv2
import os
import argparse
import boto3
import json
import base64
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# Constants
DEFAULT_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
DEFAULT_REGION = 'us-east-1'

def extract_frames(video_path, output_folder, percentage):
    """
    Trích xuất các frame từ video theo tỷ lệ phần trăm
    
    Args:
        video_path (str): Đường dẫn đến file video
        output_folder (str): Thư mục lưu các frame
        percentage (float): Tỷ lệ phần trăm frame cần trích xuất (0-100)
    """
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Đọc video
    video = cv2.VideoCapture(video_path)
    
    # Kiểm tra xem video có mở thành công không
    if not video.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return
    
    # Lấy thông tin video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Tổng số frame: {total_frames}")
    print(f"FPS: {fps}")
    
    # Tính số frame cần trích xuất
    num_frames_to_extract = int(total_frames * percentage / 100)
    print(f"Số frame sẽ trích xuất: {num_frames_to_extract}")
    
    if num_frames_to_extract <= 0:
        print("Tỷ lệ phần trăm quá nhỏ, không có frame nào được trích xuất")
        return
    
    # Tính khoảng cách giữa các frame cần trích xuất
    step = total_frames / num_frames_to_extract
    
    count = 0
    frame_index = 0
    
    # Add progress bar
    pbar = tqdm(total=num_frames_to_extract, desc="Extracting frames")
    
    while count < num_frames_to_extract:
        # Đặt vị trí đọc frame
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        
        # Đọc frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Lưu frame
        output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        count += 1
        frame_index += step
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    video.release()
    print(f"Hoàn thành! Đã trích xuất {count} frames vào thư mục {output_folder}")


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


def analyze_images_with_bedrock(image_paths, bedrock_client, model_id=DEFAULT_MODEL_ID, prompt=None):
    """
    Analyze multiple images using AWS Bedrock
    
    Args:
        image_paths (list): List of paths to image files
        bedrock_client: AWS Bedrock client
        model_id (str): Bedrock model ID to use
        prompt (str): Custom prompt for analysis
        
    Returns:
        list: List of analysis results or error messages
    """
    if not bedrock_client:
        return ["Error: No Bedrock client available"] * len(image_paths)
    
    try:
        # Convert all images to base64
        image_contents = []
        for image_path in image_paths:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
        
        # Default prompt if none provided
        if not prompt:
            prompt = "For each image, describe what you see in a concise way, focusing on the main subjects, setting, and action. Number your responses to match the order of the images."
        
        # Add text prompt as the last content item
        image_contents.append({
            "type": "text",
            "text": prompt
        })
        
        # Prepare request body
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,  # Increased token limit for multiple images
            "messages": [
                {
                    "role": "user", 
                    "content": image_contents
                }
            ]
        })

        # Call Bedrock API
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse response
        response_body = json.loads(response.get('body').read())
        analysis_text = response_body['content'][0]['text']
        
        # Split the analysis text into individual frame analyses
        # This assumes the model returns numbered responses
        analyses = []
        lines = analysis_text.split('\n')
        current_analysis = []
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if current_analysis:
                    analyses.append('\n'.join(current_analysis))
                current_analysis = [line]
            elif current_analysis:
                current_analysis.append(line)
        
        if current_analysis:
            analyses.append('\n'.join(current_analysis))
            
        # If we couldn't parse the response properly, return the whole text for each frame
        if len(analyses) != len(image_paths):
            return [analysis_text] * len(image_paths)
            
        return analyses
        
    except Exception as e:
        error_message = str(e)
        print(f"Error analyzing images: {error_message}")
        
        # Check for specific model-related errors and provide helpful message
        if "ValidationException" in error_message and "model ID" in error_message:
            available_models = ["us.amazon.nova-premier-v1:0", 
                               "us.amazon.nova-pro-v1:0",
                               "us.anthropic.claude-3-7-sonnet-20250219-v1:0"]
            return [f"Error: The specified model is not available. Try one of these models instead: {', '.join(available_models)}"] * len(image_paths)
        
        return [f"Error: {error_message}"] * len(image_paths)


def analyze_video_frames(output_folder, model_id=DEFAULT_MODEL_ID, custom_prompt=None):
    """
    Analyze extracted video frames using AWS Bedrock with Anthropic Claude model
    
    Args:
        output_folder (str): Folder containing extracted video frames
        model_id (str): Bedrock model ID to use
        custom_prompt (str): Custom prompt for analysis
        
    Returns:
        list: List of dictionaries containing frame name and analysis
    """
    # Initialize Bedrock client
    bedrock = get_bedrock_client()
    if not bedrock:
        return []
    
    # Get list of frames
    frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    frames.sort()
    
    # Get full paths for all frames
    frame_paths = [os.path.join(output_folder, frame) for frame in frames]
    
    print(f"Analyzing {len(frames)} frames in a single request...")
    
    # Analyze all frames at once
    analyses = analyze_images_with_bedrock(frame_paths, bedrock, model_id, custom_prompt)
    
    # Combine results
    results = [
        {
            'frame': frame,
            'analysis': analysis
        }
        for frame, analysis in zip(frames, analyses)
    ]
            
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze video frames")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("output_folder", help="Output folder for frames")
    parser.add_argument("percentage", type=float, help="Percentage of frames to extract (0-100)")
    parser.add_argument("--analyze", action="store_true", help="Analyze extracted frames using AWS Bedrock")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="AWS Bedrock model ID to use")
    parser.add_argument("--prompt-file", help="Path to a text file containing a custom prompt")
    
    args = parser.parse_args()
    
    # First extract frames
    extract_frames(args.video_path, args.output_folder, args.percentage)
    
    # Then analyze the extracted frames if requested
    if args.analyze:
        print("\nAnalyzing extracted frames...")
        
        # Load custom prompt if specified
        custom_prompt = None
        if args.prompt_file and os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read()
        
        results = analyze_video_frames(args.output_folder, args.model, custom_prompt)
        
        if results:
            # Save analysis results
            output_file = os.path.join(args.output_folder, 'analysis_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnalysis complete! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất frame từ video theo tỷ lệ phần trăm")
    parser.add_argument("video_path", help="Đường dẫn đến file video")
    parser.add_argument("output_folder", help="Thư mục lưu các frame")
    parser.add_argument("percentage", type=float, help="Tỷ lệ phần trăm frame cần trích xuất (0-100)")
    parser.add_argument("--analyze", action="store_true", help="Analyze extracted frames using AWS Bedrock")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="AWS Bedrock model ID to use")
    parser.add_argument("--prompt-file", help="Path to a text file containing a custom prompt")
    
    args = parser.parse_args()
    
    # Extract frames
    extract_frames(args.video_path, args.output_folder, args.percentage)
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing extracted frames...")
        
        # Load custom prompt if specified
        custom_prompt = None
        if args.prompt_file and os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read()
        
        results = analyze_video_frames(args.output_folder, args.model, custom_prompt)
        
        if results:
            # Save analysis results
            output_file = os.path.join(args.output_folder, 'analysis_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnalysis complete! Results saved to {output_file}")