from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import EmailStr

from app.api.deps import get_db, get_current_active_user
from app.db.crud import crud_user
from app.models.user import UserResponse, UserCreate, UserUpdate, UserRole

router = APIRouter()


@router.get("/", response_model=list[UserResponse])
def get_users(
        db: Session = Depends(get_db),
        current_user: UserResponse = Depends(get_current_active_user),
        skip: int = 0,
        limit: int = 100,
) -> list[UserResponse]:
    """获取用户列表（仅管理员）"""
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return crud_user.get_multi(db, skip=skip, limit=limit)


@router.get("/{email}", response_model=UserResponse)
def get_user(
        email: EmailStr,
        db: Session = Depends(get_db),
        current_user: UserResponse = Depends(get_current_active_user),
):
    """获取指定用户信息（管理员或用户本人）"""
    if current_user.role != UserRole.admin and current_user.email != email:
        raise HTTPException(status_code=403, detail="无权访问")

    db_user = crud_user.get_by_email(db, email)
    if not db_user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return db_user


@router.post("/", response_model=UserResponse)
def create_user(
        *,
        db: Session = Depends(get_db),
        user_in: UserCreate,
        current_user: UserResponse = Depends(get_current_active_user),
):
    """创建新用户（仅管理员）"""
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")

    if crud_user.get_by_email(db, email=user_in.email):
        raise HTTPException(status_code=400, detail="该邮箱已注册")

    return crud_user.create(db, user_in)


@router.put("/{email}", response_model=UserResponse)
def update_user(
        *,
        db: Session = Depends(get_db),
        email: EmailStr,
        user_in: UserUpdate,
        current_user: UserResponse = Depends(get_current_active_user),
):
    """更新用户信息（管理员或用户本人）"""
    db_user = crud_user.get_by_email(db, email)
    if not db_user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if current_user.role != UserRole.admin and current_user.email != email:
        raise HTTPException(status_code=403, detail="无权操作")

    # 如果要更新email，需要检查新email是否已存在
    if user_in.email and user_in.email != email:
        if crud_user.get_by_email(db, user_in.email):
            raise HTTPException(status_code=400, detail="新邮箱已被注册")

    return crud_user.update(db, db_user=db_user, user_in=user_in)


@router.delete("/{email}")
def delete_user(
        email: EmailStr,
        db: Session = Depends(get_db),
        current_user: UserResponse = Depends(get_current_active_user),
):
    """删除用户（仅管理员）"""
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
        
    if current_user.email == email:
        raise HTTPException(status_code=400, detail="不能删除自己的账号")
        
    user = crud_user.delete_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"msg": "用户已删除"}
