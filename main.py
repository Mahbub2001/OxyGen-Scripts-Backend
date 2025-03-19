import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime,timedelta
import uuid
from code_buddy import CodeBuddyConsole
import uvicorn
from dotenv import load_dotenv
from typing import List
from analyze_code import analyze_codes, generate_openai_response
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

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
users_collection = db.users
conversations_collection = db.conversations

code_buddy = CodeBuddyConsole()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class QueryRequest(BaseModel):
    query: str
    code: str
    scenario: str
    session_id: str = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    
class CodeAnalysisResponse(BaseModel):
    student_name: str
    file_path: str
    comments: List[str]
    rank: int
    
class UserRegister(BaseModel):
    full_name: str
    user_id: str
    email: str
    password: str

class UserInDB(BaseModel):
    user_id: str
    email: str
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str


async def get_user(user_id: str):
    user = await users_collection.find_one({"user_id": user_id})
    return UserInDB(**user) if user else None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/register")
async def register(user: UserRegister):
    existing_user = await users_collection.find_one({"user_id": user.user_id})
    if existing_user:
        raise HTTPException(status_code=400, detail="User ID already exists")

    hashed_password = get_password_hash(user.password)
    new_user = {
        "full_name": user.full_name,
        "user_id": user.user_id,
        "email": user.email,
        "hashed_password": hashed_password
    }
    await users_collection.insert_one(new_user)
    return {"message": "User registered successfully"}


@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"user_id": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user["user_id"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/protected-route")
async def protected_route(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"message": "Access granted", "user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    # print(f"Incoming request payload: {request.dict()}")
    session_id = request.session_id or str(uuid.uuid4())
    
    # print(f"Processing query for session_id: {request.query}")
    
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


@app.post("/analyze-codes")
async def analyze_codes_endpoint(files: List[UploadFile] = File(...), question: str = None):
    return await analyze_codes(files, question)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)