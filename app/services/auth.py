from typing import Optional
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.security import verify_password


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by their email and password.

    Args:
        db (Session): The database session to use for querying the user.
        email (str): The email address of the user.
        password (str): The plain text password of the user.

    Returns:
        Optional[User]: The authenticated user object if authentication is successful, None otherwise.
    """
    user: Optional[User] = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
