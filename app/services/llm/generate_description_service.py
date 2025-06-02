import os
import random
from typing import Dict, Optional
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException, Depends
from app.core.database import get_db  
from app.models.schemas import ProductData

# Models


class ProductDescriptionGenerator:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            temperature=0.1,
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            api_key=groq_api_key
        )
 


    def get_product_by_id(self, product_id: str) -> Optional[ProductData]:
        try:
            result = self.db.execute(
                text("""
                    SELECT 
                        gender,
                        mastercategory as master_category,
                        subcategory as sub_category,
                        articletype as article_type,
                        basecolour as base_color,
                        season,
                        usage,
                        productdisplayname as display_name ,
                        price ,
                        brand
                    FROM products 
                    WHERE id = :id
                """),
                {"id": str(product_id)}
            )
            

            columns = result.keys()
            product = result.fetchone()
            
            if product:
                product_dict = dict(zip(columns, product))
                
                return ProductData(
                    gender=product_dict['gender'],
                    master_category=product_dict['master_category'],
                    sub_category=product_dict['sub_category'],
                    article_type=product_dict['article_type'],
                    base_color=product_dict['base_color'],
                    season=product_dict['season'],
                    usage=product_dict['usage'],
                    display_name=product_dict['display_name'],
                    price=float(product_dict['price']), 
                    brand=product_dict['brand'] 
                )
            return None
            
        except Exception as e:
            print(f"Error fetching product: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )

    def generate_description(self, product_id: str) -> str:
        product_data = self.get_product_by_id(product_id)
        if not product_data:
            raise HTTPException(
                status_code=404,
                detail="Product not found"
            )
            
        template = """
You are an expert in creating professional and engaging product descriptions for online stores. Your task is to write a clear and well-structured paragraph that effectively describes the product using the following metadata. Make sure to highlight the key features and benefits of the product in a natural, flowing paragraph suitable for a product listing. 

The description should:
1. Be concise and user-friendly.
2. Use a professional, informative tone.
3. Be written as a cohesive paragraph rather than a list or bullet points.
4. Focus on explaining how the product's features benefit the user.
5. Start the description by the display name of the product
6. Don't necessary stick with the order of the given metadata
7.Prices are in Dollars ($)

Metadata:
{metadata}

Generated Product Description:
"""
        
        metadata_str = "\n".join([f"{key}: {value}" for key, value in product_data.dict().items()])
        print(metadata_str)
        prompt = PromptTemplate(input_variables=["metadata"], template=template)
        chain = prompt | self.llm
        return chain.invoke({"metadata": metadata_str}).content

def describe_product(product_id: str, db: Session):
    generator = ProductDescriptionGenerator(db)
    return generator.generate_description(product_id)