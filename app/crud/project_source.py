from sqlmodel import Session, select

from app.models.write_project import ProjectSourcesTable


def insert_or_update(session: Session, project_source: ProjectSourcesTable):
    # 查找是否已存在相同 project_uid 的记录
    existing = session.exec(
        select(ProjectSourcesTable).where(
            ProjectSourcesTable.project_uid == project_source.project_uid
        )
    ).first()

    if existing:
        # 如果记录存在，更新 sources
        existing.sources = project_source.sources  # 现在 sources 已经是 JSON 字符串
        session.add(existing)
    else:
        # 如果记录不存在，创建新记录
        session.add(project_source)

    # 提交事务
    session.commit()
    session.refresh(project_source)
    return project_source


def get(session: Session, project_uid):
    statement = select(ProjectSourcesTable).where(
        ProjectSourcesTable.project_uid == project_uid
    )
    return session.exec(statement).first()
