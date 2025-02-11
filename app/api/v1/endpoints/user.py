from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Form
from loguru import logger

from app.core.security import get_current_active_user, get_password_hash
from app.crud.user import update_user, insert_user, UserExistError
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


@router.post('/register')
async def user_registry(
        session: SessionDep,
        email: str = Form(...),
        username: str = Form(...),
        password: str = Form(...)
):
    user = UserTable(
        email=email,
        username=username,
        hashed_password=get_password_hash(password),
        is_active=True,
        role=UserRole.WRITER
    )

    try:
        insert_user(session, user)
        logger.info(f'创建了新用户 {username}')
        return {"message": "注册成功"}
    except UserExistError as e:
        logger.error(f'注册失败：邮箱 {email} 已存在')
        raise HTTPException(
            status_code=400,
            detail="此邮箱已被注册"
        ) from e
