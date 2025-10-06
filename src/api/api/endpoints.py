from fastapi import APIRouter, Request
import logging

from qdrant_client import QdrantClient

from api.api.models import RAGRequest, RAGResponse

from api.rag.retrieval_generation import rag_pipeline



logger = logging.getLogger(__name__)

rag_router = APIRouter()

qdrant_client = QdrantClient(url="http://qdrant:6333")

@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer = rag_pipeline(payload.query, qdrant_client)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])