import os
import json
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Load the trained model
model = load_model('tunisian_dinar_model.h5')

# Set image dimensions expected by the model
img_width, img_height = 150, 150

# Define function to preprocess a frame (NumPy array) for the model
def preprocess_frame(frame):
    # Convert BGR to RGB (since OpenCV uses BGR, but Keras expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model's input size
    frame_resized = cv2.resize(frame_rgb, (img_width, img_height))
    # Normalize pixel values to [0, 1]
    frame_normalized = frame_resized / 255.0
    # Add batch dimension
    frame_array = np.expand_dims(frame_normalized, axis=0)
    return frame_array

# Define function to predict on a frame
def predict_frame(frame):
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return class_labels[predicted_class], confidence

# Open default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize variables for prediction timing and results
last_prediction_time = time.time()
predicted_class = ""
confidence = 0.0

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Check if 1 second has passed since the last prediction
    current_time = time.time()
    if current_time - last_prediction_time >= 1:
        # Make prediction on the current frame
        predicted_class, confidence = predict_frame(frame)
        last_prediction_time = current_time

    # Draw the prediction text on the frame
    text = f"Predicted: {predicted_class} Confidence: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the prediction text
    cv2.imshow('Live Camera with Prediction', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("✅ Live prediction stopped.")