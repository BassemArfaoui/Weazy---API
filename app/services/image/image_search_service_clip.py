import uuid
from datetime import datetime
import os
import numpy as np
from PIL import Image
from app.core.utils import download_image, get_random_message
from app.core.clip_image_feature_extractor import CLIPImageFeatureExtractor
from pymilvus import Collection
from dotenv import load_dotenv

clip_fe = CLIPImageFeatureExtractor()

load_dotenv()

CLIP_THRESHOLD = float(os.getenv("CLIP_THRESHOLD", "0.1"))  # Adjust based on experiments

def handle_image_search_clip(request):
    request_data = request.dict()
    local_path = None
    distances = []

    print(request)

    try:
        # Load Milvus collection
        collection = Collection("fashion_products_image")

        # Download and process the image
        local_path = download_image(request_data["image_url"])
        query_vector = clip_fe.encode(local_path).astype('float32').reshape(1, -1)

        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 30}
        }

        # Perform the search
        results = collection.search(
            data=query_vector.tolist(),
            anns_field="clip_features",
            param=search_params,
            limit=request_data["top_k"],
            output_fields=[
                "id", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season", "usage",
                "gender", "year"
            ]
        )

        # Parse results
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

        # Final response
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

        # if distances:
        #     with open("clip_distances.txt", "a") as f:
        #         for distance in distances:
        #             f.write(f"{distance}\n")
