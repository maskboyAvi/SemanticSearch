import os
import platform

# For Windows, add the path to the ffmpeg executable (adjust this path as needed)
if platform.system() == "Windows":
    # Replace with the actual path to your ffmpeg bin directory
    ffmpeg_bin_path = r"C:\Program Files\ffmpeg\bin"
    os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

import platform
import ctypes.util

if platform.system() == "Windows":
    original_find_library = ctypes.util.find_library
    def patched_find_library(name):
        # When asked for the C library, return "msvcrt" on Windows
        if name == "c":
            return "msvcrt"
        return original_find_library(name)
    ctypes.util.find_library = patched_find_library

import cv2
import json
import numpy as np
from moviepy.editor import VideoFileClip
import torch
import faiss

# Import image captioning pipeline from transformers (using a BLIP model)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Import Whisper for audio transcription
import whisper

# Import SentenceTransformer for text embeddings
from sentence_transformers import SentenceTransformer

# Define directories for videos, frames, and outputs
VIDEOS_DIR = "videos"  # Directory where your videos are stored
# Save frames inside the static/frames folder
FRAMES_DIR = os.path.join("static", "frames")
INDEX_OUTPUT_FILE = "frame_index.faiss"
METADATA_OUTPUT_FILE = "frame_metadata.json"

# Create necessary directories if they do not exist
os.makedirs(FRAMES_DIR, exist_ok=True)

# Load models once
# Load BLIP image captioning model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Whisper model for audio transcription
whisper_model = whisper.load_model("base")

# Load SentenceTransformer model for semantic embeddings
sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_frames(video_path, frame_interval=5):
    """
    Extract frames from the video every 'frame_interval' seconds.
    Saves the frames inside the static/frames folder.
    Returns a list of dictionaries containing the frame image's relative path and its timestamp.
    """
    video_capture = cv2.VideoCapture(video_path)
    frames_info = []
    
    # Get video FPS and total frame count
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    current_time = 0
    while current_time < duration:
        # Calculate frame number for the current timestamp
        frame_number = int(current_time * fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Create a filename for the frame
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_filename = f"{video_basename}_frame_{int(current_time)}.jpg"
        # Full path where the frame will be saved
        full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
        cv2.imwrite(full_frame_path, frame)
        
        # Store a relative path with respect to the 'static' folder.
        # This will be "frames/filename.jpg"
        relative_frame_path = os.path.relpath(full_frame_path, "static")
        
        frames_info.append({
            "frame_path": relative_frame_path,
            "timestamp": current_time
        })
        current_time += frame_interval

    video_capture.release()
    return frames_info

def transcribe_audio(video_path):
    """
    Transcribes the audio from the video using Whisper.
    Returns a list of transcription segments with start and end times.
    """
    # Load the video clip and write out the audio to a temporary file
    video_clip = VideoFileClip(video_path)
    audio_temp_path = f"{os.path.splitext(video_path)[0]}_audio.mp3"
    video_clip.audio.write_audiofile(audio_temp_path, logger=None)
    
    # Use Whisper to transcribe the audio with timestamps
    result = whisper_model.transcribe(audio_temp_path)
    transcription_segments = result.get("segments", [])
    
    # Clean up the temporary audio file
    os.remove(audio_temp_path)
    return transcription_segments

def get_audio_segment_text(transcription_segments, timestamp):
    """
    Retrieves the transcription text that best matches the given timestamp.
    """
    for segment in transcription_segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        if start_time <= timestamp <= end_time:
            return segment.get("text", "").strip()
    return ""

def caption_image(image_path):
    """
    Generates a caption for the given image using the BLIP model.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def process_videos():
    """
    Processes each video in VIDEOS_DIR:
    - Extract frames (saved in static/frames)
    - Transcribe audio
    - Generate captions for frames
    - Combine caption with corresponding audio snippet
    - Compute semantic embedding for the combined text
    Returns:
        embeddings: Numpy array of embeddings for each frame
        metadata_list: List of metadata dictionaries for each frame
    """
    all_embeddings = []
    metadata_list = []
    
    # Loop over each video file in the videos directory
    for video_file in os.listdir(VIDEOS_DIR):
        video_path = os.path.join(VIDEOS_DIR, video_file)
        if not os.path.isfile(video_path):
            continue
        
        print(f"Processing video: {video_file}")
        
        # Transcribe the video's audio
        transcription_segments = transcribe_audio(video_path)
        
        # Extract frames from the video
        frames_info = extract_frames(video_path, frame_interval=5)
        
        for frame_info in frames_info:
            # Get the relative frame path (e.g., "frames/filename.jpg")
            relative_frame_path = frame_info["frame_path"]
            # Compute the full path (inside the static folder)
            full_frame_path = os.path.join("static", relative_frame_path)
            timestamp = frame_info["timestamp"]
            
            # Generate visual caption for the frame
            visual_caption = caption_image(full_frame_path)
            
            # Retrieve corresponding audio transcription based on timestamp
            audio_text = get_audio_segment_text(transcription_segments, timestamp)
            
            # Combine visual caption and audio text for a complete description
            combined_description = f"Visual: {visual_caption}. Audio: {audio_text}"
            
            # Generate semantic embedding for the combined description
            embedding = sentence_embedding_model.encode(combined_description)
            
            all_embeddings.append(embedding)
            metadata_list.append({
                "frame_path": relative_frame_path,  # e.g., "frames/filename.jpg"
                "timestamp": timestamp,
                "description": combined_description
            })
    
    return np.array(all_embeddings).astype("float32"), metadata_list

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the given embeddings.
    """
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    return index

def main():
    # Process videos to extract embeddings and metadata
    embeddings, metadata_list = process_videos()
    
    # Build FAISS index for fast similarity search
    index = build_faiss_index(embeddings)
    
    # Save the FAISS index to disk
    faiss.write_index(index, INDEX_OUTPUT_FILE)
    
    # Save the metadata mapping to disk
    with open(METADATA_OUTPUT_FILE, "w") as metadata_file:
        json.dump(metadata_list, metadata_file, indent=2)
    
    print("Index and metadata have been built and saved.")

if __name__ == "__main__":
    main()
