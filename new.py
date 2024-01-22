from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load the saved model
model = load_model('nagma_model.h5')

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Convert the image to grayscale and resize
    img = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
    img = img.resize((48, 48))  # Resize the image to match the input size of the model
    
    # Convert image to numpy array
    img_array = img_to_array(img)
    
    # Reshape the image for model input
    img_array = img_array.reshape(1, 48, 48, 1)
    
    # Normalize the image values if necessary (depends on your model)
    img_array = img_array / 255.0
    
    # Make the prediction
    prediction = model.predict(img_array)
    predicted_label = int(np.argmax(prediction))

    result = "Real" if predicted_label == 1 else "Fake"

    
    return JSONResponse(content={"predicted_label": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
