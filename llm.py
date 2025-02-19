from openai import OpenAI
import systemprompt
import config

client = OpenAI(api_key=config.API_KEY, base_url=config.CHAT_SERVER_URL)

def generate(user_input:str, temperature:int):
    messages = [
        {"role": "system", "content": systemprompt.prompt},
        {"role": "user", "content": user_input}
    ]
    
    completion = client.chat.completions.create(
        model=config.llm_model,
        messages=messages,
        max_tokens=10000,
        temperature=temperature,
        stream=False
    )
    
    return completion.choices[0].message.content