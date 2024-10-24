from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Define the input data model
class InputValues(BaseModel):
    value1: float
    value2: float
    value3: float
    value4: float
    value5: float
    value6: float
    value7: float
    value8: float
    value9: float
    value10: float

# Load the trained model
version = '0.1.0'
regressor = joblib.load(f'app/model__{version}.joblib')

# Initialize the FastAPI app
app = FastAPI()

@app.get("/", status_code=200)
async def health_check():
    """Health check endpoint."""
    return {"healthy": "yes", "version": version}

@app.post("/predict")
async def predict(inputs: InputValues):
    """Predict the output based on input values."""
    input_data = inputs.dict()
    
    # Make prediction
    result = regressor.predict([list(input_data.values())])[0]

    return {"prediction": result}
