from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import io
import cv2
import random
import os

app = FastAPI()

# Load the saved model
model = load_model('nagma_model.h5')

# Function to extract 5 random frames from a video
def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure we have enough frames in the video
    if frame_count < num_frames:
        num_frames = frame_count

    selected_frames = random.sample(range(frame_count), num_frames)
    frames = []

    for frame_number in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

# Prediction endpoint for video
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    # Save the video file temporarily
    video_contents = await file.read()
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as video_file:
        video_file.write(video_contents)

    # Extract 5 random frames from the video
    video_frames = extract_frames(video_path, num_frames=5)

    # Perform prediction for each frame
    predictions = []

    for frame in video_frames:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image
        resized_img = cv2.resize(gray_frame, (48, 48))

        # Convert image to numpy array
        img_array = img_to_array(resized_img)

        # Reshape the image for model input
        img_array = img_array.reshape(1, 48, 48, 1)  # Grayscale image, hence 1 channel

        # Normalize the image values if necessary (depends on your model)
        img_array = img_array / 255.0

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_label = int(np.argmax(prediction))
        predictions.append(predicted_label)

    # Count how many times frames are classified as "Real"
    num_real_frames = predictions.count(1)

    result = {"total_frames": len(predictions), "num_real_frames": num_real_frames}
    
    # Clean up: Remove temporary video file
    os.remove(video_path)

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
