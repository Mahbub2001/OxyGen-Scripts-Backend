import os
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from prompts import (
    CHAT_TEMPLATE, INITIAL_TEMPLATE, CORRECTION_CONTEXT,
    COMPLETION_CONTEXT, OPTIMIZATION_CONTEXT, GENERAL_ASSISTANT_CONTEXT,
    GENERATION_CONTEXT, COMMENTING_CONTEXT, EXPLANATION_CONTEXT,
    LEETCODE_CONTEXT, SHORTENING_CONTEXT,EXTRACT_BOOK_CONTEXT
)
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone
import json


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "oxygen"
index = pc.Index(index_name)
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
            "Extract From Book":EXTRACT_BOOK_CONTEXT,
            "LeetCode Solver": LEETCODE_CONTEXT,
            "Code Shortener": SHORTENING_CONTEXT
        }

        self.languages = ['Python', 'GoLang', 'TypeScript', 'JavaScript',
                          'Java', 'C', 'C++', 'C#', 'R', 'SQL']
    def retrieve_relevant_docs(self, query, top_k=5, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Retrieve relevant documents from Pinecone based on the query.
        Args:
            query (str): The query string.
            top_k (int): Number of top results to return.
            model_name (str): Name of the Sentence Transformer model to use.
        Returns:
            list: List of tuples containing (similarity, doc_id, chunk_id, chunk).
        """
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(query).tolist()

        try:
            query_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

        results = []
        for match in query_results.matches:
            similarity = match.score  
            doc_id = match.metadata.get("book_id", "unknown") 
            chunk_id = match.id  
            chunk = match.metadata.get("text", "") 
            results.append((similarity, doc_id, chunk_id, chunk))

        return results
    
    async def extract_book_and_page(self, query):
        """
        Extract the book name and page number from the user query using the OpenAI API.
        Args:
            query (str): The user query.
        Returns:
            tuple: (book_name, page_number) or (None, None) if not found.
        """        
        prompt = f"""Extract the book name and page number from this query:
        {query}
        If book name is Cbook then the page number is (query page number in the query + 40)
        Return ONLY in JSON format with these keys:
        {{
            "book_name": "filename.pdf",
            "page_number": 123
        }}
        Return null for missing values. Do NOT include any other text or formatting."""
        
        try:
            completion = client.chat.completions.create(
                model="qwen/qwq-32b:free",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1 
            )
            
            response = completion.choices[0].message.content
            print(f"Raw extraction response: {response}")  
            
            response = response.strip().replace('```json', '').replace('```', '')
            data = json.loads(response)
            
            book_name = data.get("book_name") + ".pdf"
            page_number = data.get("page_number")
            
            if book_name and page_number:
                return book_name, page_number
            return None, None
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            return None, None
        except Exception as e:
            print(f"Extraction error: {str(e)}")
            return None, None
        
    def query_page(self,book_id, page_number):
        """Query all text chunks from a specific book and page number."""
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        dummy_vector = [0.0] * model.get_sentence_embedding_dimension()
        filter = {
            'book_id': {'$eq': book_id},
            'page_number': {'$eq': page_number}
        }
        
        try:
            results = index.query(
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True,
                filter=filter
            )
            sorted_chunks = sorted(results.matches, key=lambda x: x.metadata['chunk_order'])
            return [chunk.metadata['text'] for chunk in sorted_chunks]
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

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
        docs_text=""        
        if scenario == "Extract From Book":
            book_name, page_number = await self.extract_book_and_page(query)
            
            # print(f"Extracted book name: {book_name}")  
            # print(f"Extracted page number: {page_number}")  
            if not book_name or not page_number:
                yield "Please specify the book name and page number in your query."
                return

            results = self.query_page(book_name, page_number)
            if not results:
                yield "No content found for the specified book and page."
                return

            docs_text = results
            # docs_text = "\n\n".join(results)
        else:
            results = self.retrieve_relevant_docs(query,top_k=5)
            for similarity, doc_id, chunk_id, chunk in results:
                # print(f"Similarity: {similarity:.4f}")
                # print(f"Book ID: {doc_id}")
                # print(f"Chunk ID: {chunk_id}")
                # print(f"Text: {chunk}")
                docs_text+=chunk
                
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
        print("docs_text",docs_text)
        # print(f"AI Response: {formatted_response}")
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