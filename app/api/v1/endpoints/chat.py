from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from app.core.security import get_current_active_user
from app.models.user import UserTable
from llm.core.model_core import load_glm4_flash


class ChatRequest(BaseModel):
    message: str


router = APIRouter()


@router.post("/question")
async def chat(
    request: ChatRequest,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    llm = load_glm4_flash()

    logger.info(f'{current_user.username}: {request.message}')
    async def generate():
        full_response = ""
        async for chunk in llm.astream(request.message):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        logger.info(f'AI: {full_response}')
    
    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )
