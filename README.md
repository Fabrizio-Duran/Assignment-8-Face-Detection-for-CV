# Face & Feature Detection App

A Streamlit-based web application for detecting and recognizing faces and other facial features using OpenCV.

## Features

- **Multiple Detection Options**: Detect faces, eyes, smiles, or full body in images
- **Image Upload**: Process uploaded images to detect selected features
- **Webcam Integration**: Real-time feature detection using your webcam
- **Image Comparison**: Compare features between two different images
- **Adjustable Parameters**: Customize detection settings for better results
- **Visual Feedback**: See detection results with bounding boxes and confidence scores

## Requirements

- Python 3.6+
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone this repository or download the files
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `.\venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. The app will open in your default web browser
3. Use the sidebar to adjust detection settings
4. Upload images or use your webcam for feature detection
5. Compare multiple images using the "Compare Images" tab

## Application Structure

- **Image Upload Tab**: Upload and process images to detect features
- **Webcam Tab**: Real-time feature detection using your webcam
- **Compare Images Tab**: Upload two images to compare detected features

## Customization

Adjust these parameters in the sidebar for better detection results:
- **Detection Type**: Choose what to detect (faces, eyes, smiles, bodies)
- **Min Neighbors**: Higher values result in fewer detections but more accuracy
- **Scale Factor**: Higher values mean faster processing but might miss some features
- **Minimum Size**: The smallest feature size to detect (in pixels)
- **Confidence Threshold**: Minimum confidence level to display detections
