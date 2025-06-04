import uuid
from datetime import datetime
import os
import numpy as np
from app.core.utils import download_image, get_random_message
from app.core.clip_feature_extractor import CLIPTextFeatureExtractor
from pymilvus import Collection
from dotenv import load_dotenv

fe = CLIPTextFeatureExtractor()

load_dotenv()
CLIP_THRESHOLD = float(os.getenv("CLIP_THRESHOLD", "0.1"))

def handle_text_search_clip(request):
    request_data = request.dict()
    query = request_data["text"]

    if not query:
        return {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "message": "❌ Query is empty.",
            "products": [],
            "sender": "model"
        }

    try:
        collection = Collection("fashion_products_text")

        query_vector = fe.encode(query).astype('float32').reshape(1, -1)

        query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 30}
        }

        results = collection.search(
            data=query_vector.tolist(),
            anns_field="clip_vector",
            param=search_params,
            limit=request_data.get("top_k", 5),
            output_fields=[
                "productId", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season",
                "usage", "gender", "year"
            ]
        )

        products = []
        distances = []

        for hits in results:
            for hit in hits:
                distances.append(hit.distance)
                if hit.distance >= CLIP_THRESHOLD:
                    entity = hit.entity
                    product = {
                        "id": str(entity["productId"]),
                        "gender": entity["gender"],
                        "mastercategory": entity["masterCategory"],
                        "subcategory": entity["subCategory"],
                        "articletype": entity["articleType"],
                        "basecolour": entity["baseColour"],
                        "season": entity["season"],
                        "year": entity["year"],
                        "usage": entity["usage"],
                        "productdisplayname": entity["productDisplayName"],
                        "link": entity["link"]
                    }
                    products.append(product)

        return {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "image_urls": None,
            "message": get_random_message(len(products) > 0),
            "products": products,
            "sender": "model"
        }

    finally:
        local_path = None  
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

        if distances:
            with open("clip_distances.txt", "a") as f:
                for d in distances:
                    f.write(f"{d}\n")
