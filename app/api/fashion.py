from fastapi import APIRouter, HTTPException ,Depends
from sqlalchemy.orm import Session
from app.models.schemas import  Request
from app.services.image.image_search_service_vgg16 import handle_image_search_vgg16
from app.services.image.image_search_service_resnet50 import handle_image_search_resnet
from app.services.llm.generate_description_service import describe_product
from app.core.database import get_db
from fastapi.responses import StreamingResponse
from app.services.llm.detect_tool_service import ToolRouter
from app.services.deepsearch.combined_deepsearch_service import combined_deep_search
from datetime import datetime
import uuid
from app.services.llm.respond_to_user_service import GeneralResponder
import traceback



router = APIRouter()

tool_router = ToolRouter()
responder = GeneralResponder()




@router.post("/fashion/process/{imageModel}")
def process(imageModel : str , request: Request):
    try:

        #tool detection
        if (request.tool == "none" or request.tool == "" or request.tool == None ) : 
            print("no tool")
            if (request.text != "") : 
                request.tool = tool_router.find_tool(request.text)
            elif(request.image_url != "") :
                request.tool = "search"


        #search tool
        if (request.tool == "search"):
            if (request.text !="" and request.image_url !="") : 
                return {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "id": str(uuid.uuid4()),
                "image_urls": None,
                "message": "hybrid search is coming soon",
                "products": [],
                "sender": "model"
                }
            
            elif (request.text !="") : 
                 return {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "id": str(uuid.uuid4()),
                "image_urls": None,
                "message": "text search is coming soon",
                "products": [],
                "sender": "model"
                }
            
            elif (request.image_url !="") : 
                if(imageModel == "vgg16"):
                    return handle_image_search_vgg16(request)

                elif (imageModel == "resnet50"):
                    return handle_image_search_resnet(request)

        #deepsearch tool
        elif (request.tool == "deepsearch"):
            print(request)
            if (request.text !="" and request.image_url !="") : 
                return {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "id": str(uuid.uuid4()),
                "image_urls": None,
                "message": "Deepsearch is coming soon, when using this tool multiple models will be working together to get you the best results",
                "products": [],
                "sender": "model"
                }
            
            elif (request.text !="") : 
                return {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "id": str(uuid.uuid4()),
                "image_urls": None,
                "message": "Deepsearch is coming soon, when using this tool multiple models will be working together to get you the best results",
                "products": [],
                "sender": "model"
                }
            
            elif (request.image_url !="") : 
                return {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "id": str(uuid.uuid4()),
                "image_urls": None,
                "message": "Deepsearch tool is coming soon, when using this tool multiple models will be working together to get you the best results",
                "products": [],
                "sender": "model"
                }
        
        #recommend tool
        elif (request.tool == "recommend"):
            return {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "image_urls": None,
            "message": "Recommending tool is coming soon, you can this tool to ask for recommendations based on your preferences",
            "products": [],
            "sender": "model"
            }
        
        #respond tool
        else:
            return {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4()),
            "image_urls": None,
            "message": responder.respond_to_user(request.text),
            "products": [],
            "sender": "model"
            }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fashion/description/{product_id}")
async def get_description(product_id: int, db: Session = Depends(get_db)):
    return {"description" : describe_product(product_id, db)}


# @router.get("/progress")
# async def get_progress():
#     return StreamingResponse(event_generator(), media_type="text/event-stream")





# @router.post("/search/fashion/search-by-image/vgg16")
# def search_images_vgg16(request: SearchRequest):
#     try:
#         return handle_image_search_vgg16(request)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/search/fashion/search-by-image/resnet50")
# def search_images_resnet50(request: SearchRequest):
#     try:
#         return handle_image_search_resnet(request)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))










    

