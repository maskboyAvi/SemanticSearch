import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from fastapi.staticfiles import StaticFiles

# Set the base directory to the location of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for index and metadata files
INDEX_FILE = os.path.join(BASE_DIR, "frame_index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "frame_metadata.json")

# Load the FAISS index from disk
try:
    index = faiss.read_index(INDEX_FILE)
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index: {e}")

# Load metadata mapping (frame paths, descriptions, etc.)
try:
    with open(METADATA_FILE, "r") as metadata_file:
        metadata_list = json.load(metadata_file)
except Exception as e:
    raise RuntimeError(f"Error loading metadata: {e}")

# Load the SentenceTransformer model for semantic query embeddings
sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if you want to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up absolute paths for static directory (contains both HTML and frames)
STATIC_DIR = os.path.join(BASE_DIR, "static")
# In this setup, frames are stored in a subfolder of static: "static/frames"

# Mount the static folder for frontend files (HTML, CSS, JS, and frames)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/search")
def search_frames(query: str = Query(..., description="Search query string")):
    """
    Search for video frames that are semantically similar to the provided query.
    Returns up to 4 image URLs of the most relevant video frames.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    
    # Generate the embedding for the query text
    query_embedding = sentence_embedding_model.encode(query).astype("float32")
    query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Perform similarity search using FAISS (retrieve top 4 matches)
    k = 4
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(metadata_list):
            # Convert backslashes to forward slashes
            frame_path = metadata_list[idx]["frame_path"].replace("\\", "/")
            # Remove any redundant "frames/" prefix if present
            if frame_path.lower().startswith("frames/"):
                frame_path = frame_path[len("frames/"):]
            # Construct the URL relative to the mounted /static directory
            # Since frames are stored in static/frames, the URL should be /static/frames/<filename>
            image_url = f"/static/frames/{frame_path}"
            results.append({"image_url": image_url})
    
    return JSONResponse(content={"results": results})

@app.get("/")
def serve_index():
    """
    Serves the main HTML file for the frontend.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
