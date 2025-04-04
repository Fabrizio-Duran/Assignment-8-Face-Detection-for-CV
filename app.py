import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Face & Feature Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directory for saving uploaded images if it doesn't exist
UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load pre-trained Haar cascade classifiers
def load_detectors():
    # Get the current file's directory path
    base_path = os.path.dirname(cv2.__file__)
    haar_path = os.path.join(base_path, 'data')
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_frontalface_default.xml'))
    
    # Load eye cascade
    eye_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_eye.xml'))
    
    # Load smile cascade
    smile_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_smile.xml'))
    
    # Load full body cascade
    body_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_fullbody.xml'))
    
    return {
        "face": face_cascade,
        "eye": eye_cascade,
        "smile": smile_cascade,
        "body": body_cascade
    }

# Load detectors once to avoid reloading
@st.cache_resource
def get_detectors():
    return load_detectors()

# Function to detect features in an image
def detect_features(image, detector_type, min_neighbors, scale_factor, min_size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the appropriate detector
    detectors = get_detectors()
    detector = detectors[detector_type]
    
    # Detect features
    features = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )
    
    return features

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, features, color=(0, 255, 0), thickness=2):
    image_with_boxes = image.copy()
    
    for (x, y, w, h) in features:
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, thickness)
    
    return image_with_boxes

# Function to process uploaded image
def process_image(image_upload, detector_type, min_neighbors, scale_factor, min_size, confidence):
    # Read image file buffer to a PIL image
    image = Image.open(image_upload)
    
    # Convert PIL image to numpy array (OpenCV format)
    image_np = np.array(image.convert('RGB'))
    
    # Convert from RGB to BGR (OpenCV format)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Start timer for processing time
    start_time = time.time()
    
    # Detect features in the image
    features = detect_features(image_cv, detector_type, min_neighbors, scale_factor, min_size)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Calculate confidence scores (for demonstration, using min_neighbors as a proxy)
    confidence_scores = [confidence] * len(features)
    
    # Draw bounding boxes
    image_with_boxes = draw_bounding_boxes(image_cv, features)
    
    # Convert back to RGB for display
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    return image_np, image_with_boxes_rgb, features, processing_time, confidence_scores

# Function to save uploaded images
def save_uploaded_image(image_upload, filename):
    img = Image.open(image_upload)
    img_path = os.path.join(UPLOAD_DIR, filename)
    img.save(img_path)
    return img_path

# Function to compare faces
def compare_features(image1_features, image2_features):
    # Simple comparison based on number of detected features
    if len(image1_features) == 0 or len(image2_features) == 0:
        return "No features detected in one or both images"
    
    # Count difference
    count_diff = abs(len(image1_features) - len(image2_features))
    
    if count_diff == 0:
        return "Same number of features detected in both images"
    else:
        return f"Different number of features detected: {count_diff} difference"

# Main App UI
def main():
    st.title("🔍 Feature Detection App")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Detection options
    detector_type = st.sidebar.selectbox(
        "Select what to detect:",
        ["face", "eye", "smile", "body"],
        index=0
    )
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    min_neighbors = st.sidebar.slider("Min Neighbors (Higher = less detections but more accurate)", 1, 10, 5)
    scale_factor = st.sidebar.slider("Scale Factor (Higher = faster but less accurate)", 1.1, 2.0, 1.3, 0.1)
    min_size = st.sidebar.slider("Minimum Size (px)", 10, 100, 30)
    
    # Display confidence threshold (for demonstration)
    confidence = st.sidebar.slider("Confidence Threshold (%)", 50, 100, 75) / 100.0
    
    # Additional options
    st.sidebar.subheader("Display Options")
    show_processing_time = st.sidebar.checkbox("Show Processing Time", True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam Capture", "Compare Images"])
    
    # Image upload tab
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save the uploaded image
            file_path = save_uploaded_image(uploaded_file, uploaded_file.name)
            
            # Process the image
            original_image, processed_image, features, proc_time, conf_scores = process_image(
                uploaded_file, detector_type, min_neighbors, scale_factor, min_size, confidence
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            
            with col2:
                st.subheader(f"Detected {detector_type}s")
                st.image(processed_image, use_container_width=True)
            
            # Display information
            st.subheader("Detection Results")
            st.write(f"Found {len(features)} {detector_type}(s)")
            
            if show_processing_time:
                st.info(f"Processing time: {proc_time:.4f} seconds")
            
            if show_confidence and len(features) > 0:
                st.subheader("Confidence Scores")
                for i, conf in enumerate(conf_scores):
                    st.progress(conf, text=f"{detector_type.capitalize()} {i+1}: {conf:.2f}")
    
    # Webcam tab
    with tab2:
        st.header("Webcam Capture")
        st.warning("Webcam functionality requires camera access.")
        
        run_webcam = st.button("Start Webcam Detection")
        stop_webcam = st.button("Stop Webcam")
        
        if "webcam_on" not in st.session_state:
            st.session_state.webcam_on = False
        
        if run_webcam:
            st.session_state.webcam_on = True
        
        if stop_webcam:
            st.session_state.webcam_on = False
        
        if st.session_state.webcam_on:
            # Create a placeholder for the webcam feed
            webcam_placeholder = st.empty()
            
            try:
                # Set up the webcam capture
                cap = cv2.VideoCapture(0)
                
                while st.session_state.webcam_on:
                    # Read a frame from the webcam
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture webcam feed")
                        break
                    
                    # Detect features
                    features = detect_features(frame, detector_type, min_neighbors, scale_factor, min_size)
                    
                    # Draw bounding boxes
                    frame_with_boxes = draw_bounding_boxes(frame, features)
                    
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Release the webcam
                cap.release()
            
            except Exception as e:
                st.error(f"Error accessing webcam: {e}")
    
    # Compare images tab
    with tab3:
        st.header("Compare Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload First Image")
            img1 = st.file_uploader("Choose first image...", type=["jpg", "jpeg", "png"], key="img1")
        
        with col2:
            st.subheader("Upload Second Image")
            img2 = st.file_uploader("Choose second image...", type=["jpg", "jpeg", "png"], key="img2")
        
        if img1 is not None and img2 is not None:
            # Process both images
            orig1, proc1, features1, _, _ = process_image(img1, detector_type, min_neighbors, scale_factor, min_size, confidence)
            orig2, proc2, features2, _, _ = process_image(img2, detector_type, min_neighbors, scale_factor, min_size, confidence)
            
            # Display processed images
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(proc1, caption=f"Image 1: {len(features1)} {detector_type}(s) detected", use_container_width=True)
            
            with col2:
                st.image(proc2, caption=f"Image 2: {len(features2)} {detector_type}(s) detected", use_container_width=True)
            
            # Compare features
            st.subheader("Comparison Results")
            comparison = compare_features(features1, features2)
            st.write(comparison)
            
            # Create a visual comparison
            st.subheader("Visual Comparison")
            chart_data = {
                "Image 1": len(features1),
                "Image 2": len(features2)
            }
            st.bar_chart(chart_data)

# Run the app
if __name__ == "__main__":
    main()
