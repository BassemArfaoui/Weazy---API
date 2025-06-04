from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

class CLIPImageFeatureExtractor:
    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')

    def encode(self, image_path: str):
        try:
            image = Image.open(image_path).convert('RGB')
            embedding = self.model.encode([image])[0]
            normalized_embedding = normalize([embedding], norm='l2')[0]
            return normalized_embedding

        except Exception as e:
            print(f"Error processing image '{image_path}': {str(e)}")
            return np.zeros(512)
