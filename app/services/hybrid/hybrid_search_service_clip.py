import uuid
from datetime import datetime
import os
import numpy as np
from PIL import Image
from app.core.utils import download_image, get_random_message
from app.core.clip_hybrid_exctractor import CLIPHybridFeatureExtractor
from pymilvus import Collection
from dotenv import load_dotenv

clip_fe = CLIPHybridFeatureExtractor()

load_dotenv()

CLIP_THRESHOLD = float(os.getenv("CLIP_THRESHOLD", "0.1"))  

def handle_image_search_clip_hybrid(request):
    request_data = request.dict()
    local_path = None
    distances = []

    try:
        collection = Collection("fashion_products_hybrid")

        image_url = request_data["image_url"]
        text_query = request_data.get("text", "")
        alpha = 0.4
        top_k =5

        hybrid_field = "clip_hybrid_vector"
        if alpha in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
            hybrid_field = f"clip_hybrid_vector_{int(alpha * 10)}"

        local_path = download_image(image_url)
        query_vector = clip_fe.encode(local_path, text_query, alpha=alpha).astype('float32').reshape(1, -1)

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 30}
        }

        results = collection.search(
            data=query_vector.tolist(),
            anns_field=hybrid_field,
            param=search_params,
            limit=top_k,
            output_fields=[
                "id", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season", "usage",
                "gender", "year"
            ]
        )

        products = []
        for hits in results:
            for hit in hits:
                distances.append(hit.distance)
                if hit.distance >= CLIP_THRESHOLD:
                    product_data = hit.entity
                    product = {
                        "id": str(product_data["id"]),
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
