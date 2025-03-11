from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from prompts import (
    CHAT_TEMPLATE, INITIAL_TEMPLATE, CORRECTION_CONTEXT,
    COMPLETION_CONTEXT, OPTIMIZATION_CONTEXT, GENERAL_ASSISTANT_CONTEXT,
    GENERATION_CONTEXT, COMMENTING_CONTEXT, EXPLANATION_CONTEXT,
    LEETCODE_CONTEXT, SHORTENING_CONTEXT
)
from langchain.prompts import PromptTemplate
import sqlite3
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API"),
)

def generate_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([text])[0].tolist()

class CodeBuddyConsole:
    def __init__(self):
        self.current_state = {
            'chat_history': [],
            'initial_input': "",
            'initial_context': "",
            'scenario_context': "",
            'thread_id': "",
            'docs_processed': False,
            'docs_chain': None,
            'uploaded_docs': None,
            'language': "Python",
            'scenario': "General Assistant",
            'temperature': 0.5,
            'libraries': []
        }

        self.scenario_map = {
            "General Assistant": GENERAL_ASSISTANT_CONTEXT,
            "Code Correction": CORRECTION_CONTEXT,
            "Code Completion": COMPLETION_CONTEXT,
            "Code Optimization": OPTIMIZATION_CONTEXT,
            "Code Generation": GENERATION_CONTEXT,
            "Code Commenting": COMMENTING_CONTEXT,
            "Code Explanation": EXPLANATION_CONTEXT,
            "LeetCode Solver": LEETCODE_CONTEXT,
            "Code Shortener": SHORTENING_CONTEXT
        }

        self.languages = ['Python', 'GoLang', 'TypeScript', 'JavaScript',
                          'Java', 'C', 'C++', 'C#', 'R', 'SQL']

    def retrieve_relevant_docs(self, query, top_k=3, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        conn = sqlite3.connect("embeddings.db")
        cursor = conn.cursor()

        query_embedding = generate_embeddings(query, model_name)
        query_embedding_array = np.array(query_embedding, dtype=np.float32)

        cursor.execute('''
            SELECT doc_id, chunk_id, chunk, embedding
            FROM embeddings
        ''')

        results = []
        for doc_id, chunk_id, chunk, embedding_blob in cursor.fetchall():
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = np.dot(query_embedding_array, embedding) / (np.linalg.norm(query_embedding_array) * np.linalg.norm(embedding))
            results.append((similarity, doc_id, chunk_id, chunk))

        results.sort(reverse=True, key=lambda x: x[0])

        return results[:top_k]

    async def get_conversation_history(self, session_id):
        MONGODB_CONNECTION_STRING = os.getenv("MONGO_URL")
        client = AsyncIOMotorClient(MONGODB_CONNECTION_STRING)
        db = client.codebuddy
        conversations_collection = db.conversations
        history = await conversations_collection.find({"session_id": session_id}).sort("timestamp").to_list(100)
        return [(entry['user_query'], entry['ai_response']) for entry in history]

    async def process_query_stream(self, language, code, query, scenario, session_id):
        if scenario not in self.scenario_map:
            raise ValueError(f"Invalid scenario. Choose from: {list(self.scenario_map.keys())}")

        self.current_state['scenario'] = scenario
        self.current_state['scenario_context'] = self.scenario_map[scenario]
        self.current_state['language'] = language

        history = await self.get_conversation_history(session_id)
        chat_history = self.format_conversation_history(history)

        relevant_docs = self.retrieve_relevant_docs(query)
        docs_text = "\n\n".join([doc[3] for doc in relevant_docs])

        if not history:
            prompt_template = PromptTemplate(
                input_variables=['input', 'language', 'scenario', 'scenario_context', 'code_context', 'libraries', 'docs', 'chat_history'],
                template=INITIAL_TEMPLATE
            )
        else:
            prompt_template = PromptTemplate(
                input_variables=['input', 'language', 'scenario', 'scenario_context', 'code_context', 'libraries', 'docs', 'chat_history', 'most_recent_ai_message', 'code_input'],
                template=CHAT_TEMPLATE
            )
            most_recent_ai_message = history[-1][1] if history else ""

        if not history:
            response = await self.generate_openai_response(
                prompt_template=prompt_template,
                input=code,
                code_context=query,
                language=self.current_state['language'],
                scenario=self.current_state['scenario'],
                scenario_context=self.current_state['scenario_context'],
                libraries=self.current_state['libraries'],
                docs=docs_text,
                chat_history=chat_history
            )
        else:
            response = await self.generate_openai_response(
                prompt_template=prompt_template,
                input=query,
                code_context=query,
                language=self.current_state['language'],
                scenario=self.current_state['scenario'],
                scenario_context=self.current_state['scenario_context'],
                libraries=self.current_state['libraries'],
                docs=docs_text,
                chat_history=chat_history,
                most_recent_ai_message=most_recent_ai_message,
                code_input=code
            )

        MONGODB_CONNECTION_STRING = os.getenv("MONGO_URL")
        client = AsyncIOMotorClient(MONGODB_CONNECTION_STRING)
        db = client.codebuddy
        conversations_collection = db.conversations
        await conversations_collection.insert_one({
            "session_id": session_id,
            "user_query": query,
            "ai_response": response,
            "timestamp": datetime.utcnow()
        })
        
        formatted_response = "\n".join(line for line in response.splitlines() if line.strip())
        yield formatted_response

    async def generate_openai_response(self, prompt_template, **kwargs):
        prompt = prompt_template.format(**kwargs)
        # print(prompt)
        completion = client.chat.completions.create(
            # extra_headers={
            #     "HTTP-Referer": os.getenv("YOUR_SITE_URL"),  # Optional. Site URL for rankings on openrouter.ai.
            #     "X-Title": os.getenv("YOUR_SITE_NAME"),  # Optional. Site title for rankings on openrouter.ai.
            # },
            extra_body={},
            model="qwen/qwq-32b:free", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return completion.choices[0].message.content

    def format_conversation_history(self, history):
        return "\n".join([f"User: {q}\nAI: {a}" for q, a in history])