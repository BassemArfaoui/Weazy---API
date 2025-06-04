from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

class CLIPTextFeatureExtractor:
    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')

    def encode(self, text: str):
        try:
            embedding = self.model.encode([text])[0]


            normalized_embedding = normalize([embedding], norm='l2')[0]
            
            return normalized_embedding

        except Exception as e:
            print(f"Error processing text '{text}': {str(e)}")
            return np.zeros(512) 

