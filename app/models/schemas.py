from pydantic import BaseModel

class SearchRequest(BaseModel):
    image_url: str
    chat_id: str
    sender_role: str
    top_k: int = 10
    text: str


class Request(BaseModel):
    tool : str
    image_url: str
    chat_id: str
    sender_role: str
    top_k: int = 10
    text: str

class ProductData(BaseModel):
    gender: str
    master_category: str
    sub_category: str
    article_type: str
    base_color: str
    season: str
    usage: str
    display_name: str 
    price: float
    brand: str