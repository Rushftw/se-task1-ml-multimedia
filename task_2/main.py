from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Sample model input (adjust this according to your model's input)
class ModelInput(BaseModel):
    text: str

# Sample model output (adjust this based on what your model returns)
class ModelOutput(BaseModel):
    prediction: str

# Sample route to test API
@app.post("/predict", response_model=ModelOutput)
async def predict(input: ModelInput):
    # Here, load your pre-trained model and run prediction
    # For now, this is just a dummy response
    prediction = "positive" if "good" in input.text else "negative"
    return ModelOutput(prediction=prediction)
