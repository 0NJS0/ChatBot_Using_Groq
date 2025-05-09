import os
from typing import List,Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str
    role: str = "user"
    ConvoID: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]  
        self.active: bool = True

conversations : Dict[str, Conversation] = {}


def query_groq_api(conversations: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversations.messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Groq API: {str(e)}")


def create_conversation(convo_id: str) -> Conversation:
    if convo_id not in conversations:
        conversations[convo_id] = Conversation()
    return conversations[convo_id]

@app.post("/chat")
async def chat(user_input: UserInput):
    conversation= create_conversation(user_input.ConvoID)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="Conversation is inactive. Please start a new conversation.")
    
    try:
        conversation.messages.append({"role": user_input.role, "content": user_input.message})
        response = query_groq_api(conversation)
        conversation.messages.append({"role": "assistant", "content": response})
        return {"response": response, "convo_id":user_input.ConvoID }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
