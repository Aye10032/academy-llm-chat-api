from typing import Annotated

from fastapi import APIRouter, Query

from app.crud.knowledge_base import get_knowledge_bases
from app.db.session import SessionDep
from app.schemas.knowledge_base import KnowledgeBase

router = APIRouter()


@router.get("/knowledge_bases/", response_model=list[KnowledgeBase])
def read_knowledge_bases(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    return get_knowledge_bases(session, offset, limit)
