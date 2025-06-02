import os
import numpy as np
from pymilvus import Collection
from app.core.feature_extractor_resnet import ResNetFeatureExtractor
from app.core.utils import download_image
from dotenv import load_dotenv
from PIL import Image
from app.core.feature_extractor_vgg import FeatureExtractor

load_dotenv()


vgg_fe=FeatureExtractor()
resnet_fe = ResNetFeatureExtractor()

def search_resnet_vector_with_data(request):


    load_dotenv()
    local_path = None
    collection = Collection("fashion_products")

    try:
        request_data = request.dict()
        local_path = download_image(request_data["image_url"])
        query_vector = resnet_fe.extract_features(local_path).astype("float32").reshape(1, -1)
        query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

        results = collection.search(
            data=query_vector.tolist(),
            anns_field="resnet_vector",
            param={"metric_type": "IP", "params": {"nprobe": 100}},
            limit=request_data["top_k"],
            output_fields=[
                "productId", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season",
                "usage", "gender", "year"
            ]
        )

        product_results = []
        for hits in results:
            for hit in hits:
                product = {
                    "id": str(hit.entity["productId"]),
                    "similarity": float(hit.distance),
                    "gender": hit.entity["gender"],
                    "mastercategory": hit.entity["masterCategory"],
                    "subcategory": hit.entity["subCategory"],
                    "articletype": hit.entity["articleType"],
                    "basecolour": hit.entity["baseColour"],
                    "season": hit.entity["season"],
                    "year": hit.entity["year"],
                    "usage": hit.entity["usage"],
                    "productdisplayname": hit.entity["productDisplayName"],
                    "link": hit.entity["link"]
                }
                product_results.append(product)

        return product_results

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)



def search_vgg_vector_with_data(request):


    load_dotenv()
    local_path = None
    collection = Collection("fashion_products")

    try:
        request_data = request.dict()
        local_path = download_image(request_data["image_url"])
        img = Image.open(local_path)
        query_vector = vgg_fe.extract(img).astype("float32").reshape(1, -1)
        query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

        results = collection.search(
            data=query_vector.tolist(),
            anns_field="vgg_vector",
            param={"metric_type": "IP", "params": {"nprobe": 100}},
            limit=request_data["top_k"],
            output_fields=[
                "productId", "link", "productDisplayName", "masterCategory",
                "subCategory", "articleType", "baseColour", "season",
                "usage", "gender", "year"
            ]
        )

        product_results = []
        for hits in results:
            for hit in hits:
                product = {
                    "id": str(hit.entity["productId"]),
                    "similarity": float(hit.distance),
                    "gender": hit.entity["gender"],
                    "mastercategory": hit.entity["masterCategory"],
                    "subcategory": hit.entity["subCategory"],
                    "articletype": hit.entity["articleType"],
                    "basecolour": hit.entity["baseColour"],
                    "season": hit.entity["season"],
                    "year": hit.entity["year"],
                    "usage": hit.entity["usage"],
                    "productdisplayname": hit.entity["productDisplayName"],
                    "link": hit.entity["link"]
                }
                product_results.append(product)

        return product_results

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


