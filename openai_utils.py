from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
openai_client = OpenAI()

def system_prompt(func):
    def wrapper(*args, **kwargs):
        text = func(*args, **kwargs)
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    return wrapper

def user_prompt(func):
    def wrapper(*args, **kwargs):
        text = func(*args, **kwargs)
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    return wrapper

def openai_json_response(messages, model="gpt-4o-mini", temp=1, max_tokens=1024):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    return json.loads(response.choices[0].message.content)
