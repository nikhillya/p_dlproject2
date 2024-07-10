import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("P:/potato-project/models/potato_pr5.keras")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_As_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_As_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    return{
        "class": predicted_class,
        "confidence": float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

