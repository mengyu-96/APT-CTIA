
import os
from sentence_transformers import SentenceTransformer

# Try using HF Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
save_path = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")

print(f"Downloading {model_name} from mirror to {save_path}...")

try:
    model = SentenceTransformer(model_name)
    model.save(save_path)
    print("Download success!")
except Exception as e:
    print(f"Download failed: {e}")
