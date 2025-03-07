from sqlmodel import Session

from app.models import PubMedPaperTable, PubMedReferenceTable


def insert(
    session: Session, pubmed_paper: PubMedPaperTable, reference_list: list[PubMedReferenceTable]
) -> PubMedPaperTable:
    session.add(pubmed_paper)
    for reference in reference_list:
        session.add(reference)
    session.commit()
    session.refresh(pubmed_paper)
    return pubmed_paper
