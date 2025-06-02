import os
from typing import Literal
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class ToolRouter:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            temperature=0.1,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )

        # Updated prompt: Removed slashes from endpoint names
        self.template = PromptTemplate.from_template("""
You are an intelligent assistant that classifies user requests and selects the correct tool (API endpoint) to call.

Your job is to carefully read the user's message and choose the most appropriate tool from the list below.

- Your output must be **ONLY ONE** of the following endpoint names (no explanations, just one of the following):

search  
deepsearch  
recommend  
description  
respond

Rules:
- If the input is a question or statement that does not CLEARLY fit any of the above categories, respond with "respond".
- If the user is **asking about your capabilities** (e.g. "Can you search?", "Do you support recommend?", "What tools do you have?"), respond with "respond".
- Do NOT respond with a tool name just because the input contains the name of a tool — **you must classify the intent**, not the keyword.
- Consider all tools carefully before choosing. Don't guess early based on surface patterns.
- Do NOT assume based on partial matches — always evaluate intent.

──────────────────────────────────────────────
🧠 Tool descriptions:

1. search  
Use this when the user wants to **quickly find products** based on attributes like color, category, price, or brand.  
This is a **fast search** using a single AI model (like ResNet or VGG).  
Examples:  
- "Find red sneakers"  
- "Search for blue jackets"  
- "Show me summer dresses under $50"  

2. deepsearch  
Use this when the user asks for a **more thorough or slow, high-quality search**, often implying more analysis or comparison.  
This tool uses **multiple models** and cross-verifies results for better accuracy.  
Look for keywords or tone suggesting patience or depth, like:  
"deep", "detailed", "take your time", "full search", "analyze completely", "use all models".  
Examples:  
- "Do a deep search for black leather boots"  
- "Take your time and give me the best phone options"  
- "I want a full search across all models"  

3. recommend  
Use this when the user is asking for **suggestions, alternatives, or product ideas**, usually when they’re exploring or unsure.  
This includes inspiration, seasonal picks, or alternatives.  
Examples:  
- "Suggest some ideas for summer outfits"  
- "What should I wear for a beach party?"  
- "Give me alternatives to this hoodie"  

4. description  
Use this when the user wants a **short, clear, professional description** of a specific product — like what you’d see on a store page.  
This is NOT deep analysis — it’s concise and user-friendly.  
Examples:  
- "What is this product?"  
- "Describe this item"  
- "Tell me what this jacket is like"  

5. respond  
Use this for **general conversation or questions unrelated to shopping or products**, or for questions about tools, features, or system capabilities.  
Examples:  
- "How are you?"  
- "Who made you?"  
- "What can you do?"  
- "Do you support search?"  
- "Can you recommend stuff?"  

──────────────────────────────────────────────

User input:  
{user_input}

Tool:
""")


    def find_tool(self, user_input: str) -> Literal["search", "deepsearch", "recommend", "description", "respond"]:
        chain = self.template | self.llm | StrOutputParser()
        output = chain.invoke({"user_input": user_input})
        tool = output.strip().lower()

        # Safety fallback
        valid_tools = ["search", "deepsearch", "recommend", "description", "respond"]
        if tool not in valid_tools:
            return "respond"
        return tool
