import uuid
from datetime import datetime
import numpy as np
import os
from PIL import Image
from pymilvus import Collection
from dotenv import load_dotenv
from app.core.resnet_feature_extractor import extract_resnet_features_local
from app.core.feature_extractor_resnet import ResNetFeatureExtractor
from app.core.utils import download_image , get_random_message

load_dotenv()

fe=ResNetFeatureExtractor()

RESNET_THRESHOLD = float(os.getenv("RESNET_THRESHOLD", "0.6"))


def handle_image_search_resnet(request):
    request_data = request.dict()
    local_path = None
    distances = []

    try:
        # Get the collection
        collection = Collection("fashion_products")

        # Download and process the image
        local_path = download_image(request_data["image_url"])

        # Extract and normalize ResNet features
        query_vector = fe.extract_features(local_path).astype('float32').reshape(1, -1)
        query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

        # Search in Zilliz
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 30}
        }

        results = collection.search(
            data=query_vector.tolist(),
            anns_field="resnet_vector",
            param=search_params,
            limit=request_data["top_k"],
            output_fields=[
                "productId", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season",
                "usage", "gender", "year"
            ]
        )

        # Filter and format results based on threshold
        products = []
        for hits in results:
            for hit in hits:
                distances.append(hit.distance)
                if hit.distance >= RESNET_THRESHOLD:
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
            with open("resnet_distances.txt", "a") as f:
                for distance in distances:
                    f.write(f"{distance}\n")
