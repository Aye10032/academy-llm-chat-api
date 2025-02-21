from sqlmodel import Session, select

from app.models.write_project import ChatRecordTable


def insert(session: Session, chat_record: ChatRecordTable):
    session.add(chat_record)
    session.commit()
    session.refresh(chat_record)
    return chat_record


def get_by_project(session: Session, project_uid: str, offset: int, limit: int):
    statement = (
        select(ChatRecordTable)
        .where(ChatRecordTable.project_uid == project_uid)
        .offset(offset)
        .limit(limit)
    )
    return session.exec(statement).all()


def get_by_user(session: Session, user_email: str, offset: int, limit: int):
    statement = (
        select(ChatRecordTable)
        .where(ChatRecordTable.user_email == user_email)
        .offset(offset)
        .limit(limit)
    )
    return session.exec(statement).all()
