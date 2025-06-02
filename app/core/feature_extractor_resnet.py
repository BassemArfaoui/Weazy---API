import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize

class ResNetFeatureExtractor:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def extract_features(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize and preprocess image
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract and normalize features
            features = self.model.predict(x, verbose=0).flatten()
            return normalize([features], norm='l2')[0] 
        
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return np.zeros(2048)