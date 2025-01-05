from typing import Annotated

from fastapi import APIRouter, Query, Depends
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.security import get_current_active_user
from app.crud.knowledge_base import get_knowledge_bases
from app.db.session import SessionDep
from app.models import UserTable
from app.schemas.knowledge_base import KnowledgeBase
from llm.core.model_core import load_glm4_flash


class ChatRequest(BaseModel):
    message: str
    knowledge_base_name: str
    history: str


router = APIRouter()


@router.get('/knowledge_bases', response_model=list[KnowledgeBase])
async def read_knowledge_bases(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    return get_knowledge_bases(session, offset, limit)


@router.post('/chat')
async def chat(
        request: ChatRequest,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    llm = load_glm4_flash()

    logger.info(f'{current_user.username}: {request.message}')

    async def generate():
        full_response = ''
        async for chunk in llm.astream(request.message):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        logger.info(f'AI: {full_response}')

    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )
