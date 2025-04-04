# Face & Feature Detection Application

## Overview
This application uses computer vision techniques to detect faces and facial features in images and live webcam feeds. Built with Streamlit and OpenCV, it provides an interactive interface for experimenting with different detection parameters.

## Features
- **Multiple Detection Types**: Detect faces, eyes, smiles, and full bodies
- **Image Upload**: Process and analyze any uploaded image
- **Webcam Integration**: Real-time detection using your webcam
- **Image Comparison**: Compare feature detection between two different images
- **Adjustable Parameters**: Fine-tune the detection with customizable settings
- **Performance Metrics**: View processing times and confidence scores

## Requirements
- Python 3.8+
- Streamlit 1.44.1
- OpenCV 4.11.0
- NumPy 2.2.4
- Pillow 11.1.0

## Installation
1. Clone this repository:
   ```
   git clone <repository-url>
   cd Assignment-8-Face-Detection-for-CV
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # Mac/Linux
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser
   
3. Use the sidebar to adjust detection parameters:
   - Select what to detect (face, eye, smile, body)
   - Adjust Min Neighbors (higher = fewer detections but more accurate)
   - Set Scale Factor (higher = faster but less accurate)
   - Define Minimum Size for detection
   - Set Confidence Threshold

4. Switch between tabs for different functionalities:
   - **Image Upload**: Upload an image for feature detection
   - **Webcam Capture**: Use your webcam for real-time detection
   - **Compare Images**: Upload two images to compare detection results

## How it Works
The application uses pre-trained Haar cascade classifiers from OpenCV to detect various features in images. These classifiers are based on machine learning models trained to identify specific patterns in images that correspond to faces, eyes, etc.

## Project Structure
- `app.py`: Main application file with all the detection logic and UI
- `requirements.txt`: Lists all required Python packages
- `uploaded_images/`: Directory that stores user-uploaded images