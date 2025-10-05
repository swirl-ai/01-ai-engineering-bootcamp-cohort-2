from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.core.config import config
from src.api.api.endpoints import api_router

import logging
from src.api.api.middleware import RequestIDMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "API"}