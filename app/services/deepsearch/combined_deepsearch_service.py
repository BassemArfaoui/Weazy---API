import uuid
import math
from datetime import datetime
from app.core.utils import get_random_message
from app.services.deepsearch.image_deepsearch_service import (
    search_resnet_vector_with_data,
    search_vgg_vector_with_data
)

THRESHOLD = 0.6

def combined_deep_search(request, vgg_weight=0.5, resnet_weight=0.5):
    combined_scores = {}
    common_results = []
    unique_results = []
    try:
        # Validate weights
        if not (0 <= vgg_weight <= 1 and 0 <= resnet_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if not math.isclose(vgg_weight + resnet_weight, 1.0, rel_tol=1e-5):
            raise ValueError("Weights must sum to 1.0")

        request.top_k = request.top_k * 2 
        resnet_results = search_resnet_vector_with_data(request)
        print("finished resnet")
        vgg_results = search_vgg_vector_with_data(request)
        print("finished vgg")

        resnet_dict = {item["id"]: item for item in resnet_results}
        vgg_dict = {item["id"]: item for item in vgg_results}

        common_ids = set(resnet_dict) & set(vgg_dict)
        unique_resnet_ids = set(resnet_dict) - common_ids
        unique_vgg_ids = set(vgg_dict) - common_ids

        # Process common items (present in both models)
        for id_ in common_ids:
            r_score = resnet_dict[id_]["similarity"]
            v_score = vgg_dict[id_]["similarity"]
            
            weighted_avg = (r_score * resnet_weight) + (v_score * vgg_weight)
            
            if weighted_avg >= THRESHOLD:
                source_dict = resnet_dict if r_score > v_score else vgg_dict
                common_results.append({
                    "product": source_dict[id_],
                    "similarity": weighted_avg,
                    "resnet_score": r_score,
                    "vgg_score": v_score,
                    "is_common": True
                })

        # Process unique ResNet items
        for id_ in unique_resnet_ids:
            r_score = resnet_dict[id_]["similarity"]
            if r_score >= THRESHOLD:
                unique_results.append({
                    "product": resnet_dict[id_],
                    "similarity": r_score,
                    "resnet_score": r_score,
                    "vgg_score": 0,
                    "is_common": False
                })

        # Process unique VGG items
        for id_ in unique_vgg_ids:
            v_score = vgg_dict[id_]["similarity"]
            if v_score >= THRESHOLD:
                unique_results.append({
                    "product": vgg_dict[id_],
                    "similarity": v_score,
                    "resnet_score": 0,
                    "vgg_score": v_score,
                    "is_common": False
                })

        # Sort common results by weighted score (descending)
        common_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Sort unique results by their individual score (descending)
        unique_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Combine results (common first, then unique)
        top_results = common_results + unique_results
        top_results = top_results[: (request.top_k // 2)]
        print("finished comparing")

        response = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "image_urls": None,
            "message": get_random_message(len(top_results) > 0),
            "fusion_method": f"weighted_common_first(vgg={vgg_weight}, resnet={resnet_weight})",
            "products": [
                {
                    "id": item["product"]["id"],
                    "gender": item["product"]["gender"],
                    "mastercategory": item["product"]["mastercategory"],
                    "subcategory": item["product"]["subcategory"],
                    "articletype": item["product"]["articletype"],
                    "basecolour": item["product"]["basecolour"],
                    "season": item["product"]["season"],
                    "year": item["product"]["year"],
                    "usage": item["product"]["usage"],
                    "productdisplayname": item["product"]["productdisplayname"],
                    "link": item["product"]["link"],
                }
                for item in top_results
            ],
            "sender": "model",
        }

        return response

    finally:
        try:
            with open("deepsearch_distances.txt", "a") as f:
                for item in top_results:
                    pid = item["product"]["id"]
                    score = item["similarity"]
                    r_score = item.get("resnet_score", "N/A")
                    v_score = item.get("vgg_score", "N/A")
                    common_flag = "C" if item.get("is_common", False) else "U"
                    f.write(f"{pid}: {score:.4f} (R:{r_score:.4f}, V:{v_score:.4f}, {common_flag})\n")
        except Exception as e:
            print(f"[!] Failed to write similarity scores: {e}")
    combined_scores = {}
    top_results = []
    try:
        # Validate weights
        if not (0 <= vgg_weight <= 1 and 0 <= resnet_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if not math.isclose(vgg_weight + resnet_weight, 1.0, rel_tol=1e-5):
            raise ValueError("Weights must sum to 1.0")

        request.top_k = request.top_k * 2 
        resnet_results = search_resnet_vector_with_data(request)
        print("finished resnet")
        vgg_results = search_vgg_vector_with_data(request)
        print("finished vgg")

        resnet_dict = {item["id"]: item for item in resnet_results}
        vgg_dict = {item["id"]: item for item in vgg_results}

        all_ids = set(resnet_dict) | set(vgg_dict)

        for id_ in all_ids:
            # Get scores (0 if not present in one model's results)
            r_score = resnet_dict.get(id_, {}).get("similarity", 0)
            v_score = vgg_dict.get(id_, {}).get("similarity", 0)
            
            # Calculate weighted average
            weighted_avg = (r_score * resnet_weight) + (v_score * vgg_weight)
            
            if weighted_avg >= THRESHOLD:
                # Use whichever result has higher individual score as the product data
                source_dict = resnet_dict if r_score > v_score else vgg_dict
                combined_scores[id_] = {
                    "product": source_dict[id_],
                    "similarity": weighted_avg,
                    "resnet_score": r_score,
                    "vgg_score": v_score
                }

        sorted_combined = sorted(combined_scores.values(), 
                               key=lambda x: x["similarity"], 
                               reverse=True)
        top_results = sorted_combined[: (request.top_k // 2)]
        print("finished comparing")

        response = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "image_urls": None,
            "message": get_random_message(len(top_results) > 0),
            "fusion_method": f"weighted_average(vgg={vgg_weight}, resnet={resnet_weight})",
            "products": [
                {
                    "id": item["product"]["id"],
                    "gender": item["product"]["gender"],
                    "mastercategory": item["product"]["mastercategory"],
                    "subcategory": item["product"]["subcategory"],
                    "articletype": item["product"]["articletype"],
                    "basecolour": item["product"]["basecolour"],
                    "season": item["product"]["season"],
                    "year": item["product"]["year"],
                    "usage": item["product"]["usage"],
                    "productdisplayname": item["product"]["productdisplayname"],
                    "link": item["product"]["link"],
                }
                for item in top_results
            ],
            "sender": "model",
        }

        return response

    finally:
        try:
            with open("deepsearch_distances.txt", "a") as f:
                for item in top_results:
                    pid = item["product"]["id"]
                    score = item["similarity"]
                    r_score = item.get("resnet_score", "N/A")
                    v_score = item.get("vgg_score", "N/A")
                    f.write(f"{pid}: {score:.4f} (R:{r_score:.4f}, V:{v_score:.4f})\n")
        except Exception as e:
            print(f"[!] Failed to write similarity scores: {e}")