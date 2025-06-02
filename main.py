from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import init_db
from app.core.config import connectToMilvus
from app.api.fashion import router as fashion_router
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    connectToMilvus(os.getenv("ZILLIZ_URI"), os.getenv("ZILLIZ_TOKEN"))
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fashion_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)