from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import detect_intent
import uvicorn
from typing import Literal, Dict, Any

app = FastAPI()

# Define a Pydantic model to handle the request body
class QuestionRequest(BaseModel):
    question: str
    version: Literal['11.x', '10.x', '9.x']  # Replace these values with your actual versions

# Define a Pydantic model for the successful response
class IntentResponse(BaseModel):
    version: Literal['11.x', '10.x', '9.x']
    question: str
    answer: str  

# Define a Pydantic model for error responses
class ErrorResponse(BaseModel):
    detail: str

@app.post("/get_intent", response_model=IntentResponse, responses={500: {"model": ErrorResponse}})
async def get_intent(request: QuestionRequest):
    question = request.question
    version = request.version
    
    try:
        response = detect_intent.detect_intent_with_context(question, version)
        # Assume response is a dictionary with 'intent' and other relevant information
        return IntentResponse(version=version,question=question, answer=response)
    except Exception as e:
        # Returning a detailed error response
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server with Uvicorn
    uvicorn.run('app:app', host="0.0.0.0", port=8080, reload=True)
