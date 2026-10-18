import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

res = client.models.embed_content(
    model="gemini-embedding-001",
    contents="יש לי נזילה בבית, איך מגישים תביעה?",
)
vec = res.embeddings[0].values
print("OK, vector size:", len(vec))