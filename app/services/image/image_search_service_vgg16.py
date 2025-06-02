import uuid
from datetime import datetime
import os
import numpy as np
from PIL import Image
from app.core.utils import download_image , get_random_message
from app.core.feature_extractor_vgg import FeatureExtractor
from pymilvus import Collection
from dotenv import load_dotenv


fe=FeatureExtractor()

load_dotenv()

VGG_THRESHOLD = float(os.getenv("VGG_THRESHOLD", "0.6"))  



def handle_image_search_vgg16(request):
    request_data = request.dict()
    local_path = None

    print(request)
    
    try:
        
        # Get the collection
        collection = Collection("fashion_products")

        # Download and process the image
        local_path = download_image(request_data["image_url"])
        img = Image.open(local_path)
        
        # Extract and normalize 
        query_features = fe.extract(img).astype('float32').reshape(1, -1)
        query_features /= np.linalg.norm(query_features, axis=1, keepdims=True) 
        
        # Search in Zilliz
        search_params = {
            "metric_type": "IP",  
            "params": {"nprobe": 30}  
        }
        
        results = collection.search(
            data=query_features.tolist(),
            anns_field="vgg_vector",
            param=search_params,
            limit=request_data["top_k"],
            output_fields=["productId", "link", "productDisplayName", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage", "gender", "year"]
        )
        
        # Map the results
        products = []
        distances = [] 
        for hits in results:
            for hit in hits:
                distances.append(hit.distance)
                if hit.distance >= VGG_THRESHOLD:
                    product_data = hit.entity

                    
                    product = {
                        "id": str(product_data["productId"]),
                        "gender": product_data["gender"],
                        "mastercategory": product_data["masterCategory"],
                        "subcategory": product_data["subCategory"],
                        "articletype": product_data["articleType"],
                        "basecolour": product_data["baseColour"],
                        "season": product_data["season"],
                        "year": product_data["year"],
                        "usage": product_data["usage"],
                        "productdisplayname": product_data["productDisplayName"],
                        "link": product_data["link"]
                    }
                    
                    products.append(product)
        
        # Create the response
        result_message = {
            "created_at": datetime.utcnow().isoformat() + "Z", 
            "id": str(uuid.uuid4()),  
            "image_urls": None,  
            "message": get_random_message(len(products) > 0), 
            "products": products,  
            "sender": "model" 
        }
        
        return result_message
        
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

        if distances:
            with open("vgg_distances.txt", "a") as f:
                for distance in distances:
                    f.write(f"{distance}\n")
