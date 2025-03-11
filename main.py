from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import uuid
from code_buddy import CodeBuddyConsole
import uvicorn
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# MongoDB connection
MONGODB_CONNECTION_STRING = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(MONGODB_CONNECTION_STRING)
db = client.codebuddy
conversations_collection = db.conversations

code_buddy = CodeBuddyConsole()

class QueryRequest(BaseModel):
    query: str
    code: str
    scenario: str
    session_id: str = None

class QueryResponse(BaseModel):
    response: str
    session_id: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    print(f"Processing query for session_id: {request.query}")
    
    response_generator = code_buddy.process_query_stream(
        language="c", 
        code=request.code,
        query=request.query,
        scenario=request.scenario,
        session_id=session_id
    )
    
    response = ""
    async for chunk in response_generator: 
        response += chunk

    conversation = {
        "session_id": session_id,
        "user_query": request.query,
        "ai_response": response,
        "timestamp": datetime.utcnow()
    }
    await conversations_collection.insert_one(conversation)

    return QueryResponse(response=response, session_id=session_id)
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = await conversations_collection.find({"session_id": session_id}).sort("timestamp").to_list(100)
    return {"history": history}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)