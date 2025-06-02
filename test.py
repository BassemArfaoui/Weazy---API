import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import Dict
import psycopg2

# Load environment variables
load_dotenv()

host = os.getenv("PG_HOST")
database = os.getenv("PG_DATABASE")
user = os.getenv("PG_USER")
password = os.getenv("PG_PASSWORD")
port = os.getenv("PG_PORT")

# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    print("Successfully connected to the database!")
    
    # Create a cursor object to interact with the database
    

except Exception as e:
    print(f"Error: {e}")


def get_product_by_id(product_id: int):
    """Fetch product data from the database using the product id."""
    cur = None  # Initialize cur to None to ensure it's only closed if it was successfully created
    try:
        # Create a cursor object to interact with the database
        cur = conn.cursor()
        
        # Query to retrieve the product by id
        query = "SELECT * FROM products WHERE id = %s;"
        
        # Execute the query
        cur.execute(query, (str(product_id),))
        
        # Fetch the result
        product = cur.fetchone()
        
        # Check if the product was found
        if product:
            # Assuming the product table has columns like: id, gender, master_category, sub_category, article_type, base_color, season, year, usage, price, brand, etc.
            print(product)
            product_data = {
                "gender": product[1],
                "master_category": product[2],
                "sub_category": product[3],
                "article_type": product[4],
                "base_color": product[5],
                "season": product[6],
                "usage": product[8],
                "price": 5000,  # Assuming price is the 10th column in the table
                "brand": "Nike",  # Replace with actual brand data if available
                # Add more fields as needed based on the product table structure
            }

            return product_data
        else:
            return None
        
    except Exception as e:
        return f"Error: {e}"
    finally:
        if cur:
            # Close the cursor only if it was successfully created
            cur.close()
        conn.close()


# Initialize LLMs
def initialize_llms():
    """Initialize and return the LLM instances"""
    groq_api_key = os.getenv("GROQ_API_KEY")

    return {
        "llm": ChatGroq(
            temperature=0.1, model="llama-3.3-70b-versatile", api_key=groq_api_key
        ),
        "description_llm": ChatGroq(
            temperature=0.1, model="Gemma2-9B-IT", api_key=groq_api_key
        ),
    }

# Product description generation
def generate_product_description(metadata: Dict[str, str], llm: ChatGroq) -> str:
    """Generate a product description for a product based on the given metadata"""
    template = """
You are an expert in creating professional and engaging product descriptions for online stores. Your task is to write a clear and well-structured paragraph that effectively describes the product using the following metadata. Make sure to highlight the key features and benefits of the product in a natural, flowing paragraph suitable for a product listing. 

The description should:
1. Be concise and user-friendly.
2. Use a professional, informative tone.
3. Be written as a cohesive paragraph rather than a list or bullet points.
4. Focus on explaining how the product's features benefit the user.

Metadata:
{metadata}

Generated Product Description:
"""

    # Format the metadata as a string
    metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])

    prompt = PromptTemplate(input_variables=["metadata"], template=template)

    # Create chain and invoke
    chain = prompt | llm
    return chain.invoke({"metadata": metadata_str}).content

# Example usage
def describe_product(metadata: Dict[str, str]) -> str:
    """Generate a product description using the LLM"""
    llms = initialize_llms()
    description = generate_product_description(metadata, llms['description_llm'])
    return description

id = int(input("Enter the product id: "))

data = get_product_by_id(id)

# Get product description
if (data) :
  product_description = describe_product( data)
  print("Generated Product Description:")
  print(product_description)
else : 
      print("No data for this product")

