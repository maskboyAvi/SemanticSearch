# Video Frame Semantic Search

This repository contains a Python-based API that enables semantic search on a video library. The application extracts key frames from videos, creates embeddings for the frames, and then allows users to search for relevant video frames using natural language queries. The API returns up to 4 image URLs corresponding to the most relevant video frames.

## Overview

This project allows users to search for specific frames in a video library based on semantic similarity. Instead of generating images with AI, the system retrieves existing video frames that match a user query. It is built with FastAPI for the backend and uses tools like FAISS for efficient similarity search and SentenceTransformer for semantic embeddings.

---

## Features

- **Frame Extraction**: Extracts frames from videos at regular intervals.
- **Audio Transcription & Captioning**: Uses Whisper and BLIP models to capture both visual and audio context from each frame.
- **Semantic Embedding**: Combines visual and audio data into a unified description and generates embeddings using SentenceTransformer.
- **Fast Similarity Search**: Leverages FAISS to quickly search through high-dimensional embeddings.
- **REST API**: A FastAPI-based backend that exposes a `/search` endpoint to query the video frames.
- **Simple Frontend**: An HTML-based interface (served from the `static` directory) for easy interaction.

---

## Installation

### For Full Embedding & Indexing (Processing Videos)

If you want to process your video library to extract frames, transcribe audio, generate captions, and build the FAISS index (i.e., run the full embedding pipeline):

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create & Activate a Virtual Environment**:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install All Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place Your Videos**:
   Put your sample videos (e.g., the Ember.zip sample data) in the `videos` folder.

5. **Run the Video Processing Script**:
   This script extracts frames and builds the semantic index.
   ```bash
   python video_processing.py
   ```

This will create the following output files:
- `frame_index.faiss` – The FAISS index of embeddings.
- `frame_metadata.json` – Metadata with paths (stored as relative paths under `static/frames`).

### For Testing the API Only

If you just want to test the API (assuming the frames have already been processed and the index exists):

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create & Activate a Virtual Environment** (optional but recommended):
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Minimal Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure that `frame_index.faiss` and `frame_metadata.json` are already present in the repository. You can use pre-built files if provided.)*

4. **Start the FastAPI App**:
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the Application**:
   Open your browser and navigate to [http://localhost:8000/](http://localhost:8000/) to see the search interface.

---

## Usage

- **Search Endpoint**:  
  - URL: `/search`
  - Method: GET
  - Query Parameter: `query` (e.g., `ember`)
  - Response: JSON containing up to 4 image URLs for the most relevant video frames.
  
  **Example Request**:
  ```
  GET /search?query=ember
  ```

  **Example Response**:
  ```json
  {
    "results": [
      {"image_url": "/static/frames/MTX_2021_P001 Meet Ember_frame_0.jpg"},
      {"image_url": "/static/frames/MTX_2021_P001 Meet Ember_frame_5.jpg"}
    ]
  }
  ```

- **Frontend**:  
  The homepage served at `/` (i.e., [http://localhost:8000/](http://localhost:8000/)) provides a simple search box to enter queries and display results.

---

## Design Decisions & Approach

The design of this project was driven by the need for efficiency and clarity:

- **Framework Choice**: FastAPI was chosen because of its performance, modern Python features, and ease of integrating with ML libraries.
- **Semantic Embedding**: Instead of relying solely on visual features, the approach combines audio transcription and image captioning to form a richer description of each video frame. This allows for more accurate semantic searches.
- **Similarity Search**: FAISS was implemented to quickly search through the high-dimensional embedding space, making the system scalable even with larger video libraries.
- **Static File Serving**: Extracted frames are stored under `static/frames`, so they can be easily served by FastAPI. This simplifies the URL management for images.
- **Dual Installation Options**: The repository is structured to allow users to either process videos (for embedding/index generation) or simply run the pre-processed API, catering to different user needs.

---

## Project Structure

```
your_project/
├── app.py                    # FastAPI application
├── video_processing.py       # Script to process videos and build embeddings/index
├── frame_index.faiss         # FAISS index (generated by video_processing.py)
├── frame_metadata.json       # Metadata mapping for frames (generated by video_processing.py)
├── requirements.txt          # Python dependencies
├── videos/                   # Folder containing video files for processing
└── static/                   # Static files served by the app
    ├── index.html          <-- Frontend HTML file
    └── frames/             <-- Extracted video frames are saved here
```

---

## License

This project is released under the [MIT License](LICENSE).

---

Feel free to reach out with any questions or suggestions. Enjoy building and testing the app!