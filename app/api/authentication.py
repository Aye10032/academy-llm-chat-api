from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from typing import Optional

from app.core.config import settings
from app.models.user import User
from app.db.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")


async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Retrieve the current user based on the provided JWT token.

    Args:
        token (str): The JWT token provided by the client.
        db (Session): The database session to use for querying the user.

    Returns:
        Optional[User]: The authenticated user object if the token is valid, None otherwise.

    Raises:
        HTTPException: If the token is invalid or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user: Optional[User] = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user
