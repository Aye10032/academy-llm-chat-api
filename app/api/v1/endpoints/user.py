from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_current_active_user
from app.crud.user import update_user
from app.db.session import SessionDep
from app.models import UserTable
from app.schemas.user import UserPublic, UserUpdate, UserRole

router = APIRouter()


@router.patch('/{email}', response_model=UserPublic)
async def update_chat(
        session: SessionDep,
        email: str,
        user: UserUpdate,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    if not (email == current_user.email or current_user.role >= UserRole.ADMIN):
        raise HTTPException(status_code=11, detail='您无权修改此用户的信息！')

    return update_user(session, email, user)
