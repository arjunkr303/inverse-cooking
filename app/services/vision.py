import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("WARNING: 'HF_TOKEN' is not set in your .env file. API requests may fail.")

client = InferenceClient(api_key=hf_token)

async def query_qwen_vision(image_base64: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Run the synchronous InferenceClient call in a thread pool to avoid blocking the async event loop
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        max_tokens=800
    )
    
    return response.choices[0].message.content

async def generate_recipe_from_image(image_base64: str) -> str:
    """
    Analyze a food image (base64) and generate a detailed recipe using Qwen2.5-VL.
    """
    prompt = (
        "Analyze this food image and generate a detailed recipe. "
        "Include: 1. Recipe Name, 2. Prep & Cook Time, 3. Ingredients with measurements, "
        "4. Step-by-step Instructions."
    )
    return await query_qwen_vision(image_base64, prompt)
