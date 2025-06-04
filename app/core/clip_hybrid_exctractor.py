from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

class CLIPHybridFeatureExtractor:
    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')

    def encode(self, image_path: str, text: str, alpha: float = 0.5):
        """
        Compute hybrid embedding from image and text.
        alpha: weight for the image (between 0 and 1)
        (1 - alpha): weight for the text
        """
        try:
            # Encode image
            image = Image.open(image_path).convert('RGB')
            image_emb = self.model.encode([image])[0]

            # Encode text
            text_emb = self.model.encode([text])[0]

            # Weighted average
            hybrid_emb = alpha * image_emb + (1 - alpha) * text_emb

            # Normalize
            normalized_emb = normalize([hybrid_emb], norm='l2')[0]
            return normalized_emb

        except Exception as e:
            print(f"Error processing image/text ({image_path}, '{text}'): {str(e)}")
            return np.zeros(512)
