from fastapi import HTTPException, UploadFile
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API"),
)

async def analyze_codes(files: List[UploadFile], question: str = None) -> List[Dict]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    analysis_results = []

    for file in files:
        code_content = await file.read()
        code_content = code_content.decode("utf-8")
        
        comments = await generate_openai_response(
            "Analyze the following C code and provide feedback in the following format:\n\n"
            "1. Original student code with comments indicating improvements or errors.\n"
            "2. A perfect version of the code (if applicable).\n"
            "3. Explanations for the improvements or errors in a comment block.\n\n"
            "Format the response as follows:\n"
            "\n"
            "// Original student code with comments\n"
            "<original code with comments>\n\n"
            "// Perfect code\n"
            "<perfect code>\n\n"
            "/* \n"
            "Explanation of improvements or errors:"
            "<explanations>\n"
            "\n\n"
            "Here is the code to analyze:\n\n"
            "{code}\n\n"
            "Question: {question}"
            "*/\n",
            code=code_content,
            question=question
        )
        rank = len(comments.split("\n")) 
        
        analysis_results.append({
            "student_name": file.filename,
            "file_content": code_content,  
            "comments": comments.split("\n"),
            "rank": rank
        })
    
    return analysis_results

async def generate_openai_response(prompt_template: str, **kwargs) -> str:
    prompt = prompt_template.format(**kwargs)
    completion = client.chat.completions.create(
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