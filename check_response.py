import google.genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = google.genai.Client(api_key=api_key)

try:
    print("Trying to generate content...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=["Hello, tell me a short joke"]
    )
    print("Response type:", type(response))
    print("Response attributes:", dir(response))
    
    if hasattr(response, 'text'):
        print("Response has text attribute:", response.text)
    elif hasattr(response, 'content'):
        print("Response has content attribute:", response.content)
    elif hasattr(response, 'parts'):
        print("Response has parts attribute:", response.parts)
    else:
        print("Response keys:", [attr for attr in dir(response) if not attr.startswith('_')])
        
except Exception as e:
    print(f'Error: {e}')
