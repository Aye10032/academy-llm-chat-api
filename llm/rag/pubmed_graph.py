from datetime import date

from app.core.config import get_settings
from llm.rag.graph_storage import NebulaGraphStore
from llm.schemas.nebula_graph import Edge, Prop, PropType, Tag, VidType
from llm.schemas.pubmed_data import PubMedData

nebula_cfg = get_settings().knowledge_base.nebula


def init_pubmed_graph(drop_old: bool = False):
    with NebulaGraphStore(
        address=nebula_cfg.HOST,
        port=nebula_cfg.PORT,
        username=nebula_cfg.USERNAME,
        password=nebula_cfg.PASSWORD,
    ) as store:
        if drop_old:
            store.drop_space('pubmed', check_exist=True)

        result = store.create_space('pubmed', vid_type=VidType.STRING255)
        assert result.is_succeeded(), result.error_msg()

        store.use_space('pubmed')

        store.create_tag(
            Tag(
                tag_name='Paper',
                props=[
                    Prop[str](
                        prop_name='pmid',
                        data_type=PropType.STRING,
                        comment='文献的pubmed编号',
                        not_null=True,
                    ),
                    Prop[str](
                        prop_name='title',
                        data_type=PropType.STRING,
                        comment='文献的标题',
                        not_null=True,
                    ),
                    Prop[date](
                        prop_name='pub_date',
                        data_type=PropType.DATE,
                        comment='文章的发表日期',
                        not_null=True,
                    ),
                    Prop[str](prop_name='doi', data_type=PropType.STRING, comment='文献的doi编号'),
                    Prop[str](
                        prop_name='abstract', data_type=PropType.STRING, comment='文献的摘要信息'
                    ),
                    Prop[int](
                        prop_name='reference_num',
                        data_type=PropType.INT64,
                        comment='文献的被引次数',
                    ),
                ],
            ),
            check_exist=True,
        )
        store.create_tag(
            Tag(
                tag_name='Author',
                props=[
                    Prop[str](
                        prop_name='name',
                        data_type=PropType.STRING,
                        comment='作者姓名',
                        not_null=True,
                    ),
                    Prop[str](
                        prop_name='affiliation',
                        data_type=PropType.STRING,
                        comment='作者所属的机构信息',
                    ),
                ],
                comment='文献的作者信息',
            ),
            check_exist=True,
        )
        store.create_tag(
            Tag(
                tag_name='Keyword',
                props=[
                    Prop[str](
                        prop_name='text',
                        data_type=PropType.STRING,
                        not_null=True,
                        comment='关键词内容',
                    ),
                ],
                comment='文献的关键词',
            ),
            check_exist=True,
        )
        store.create_tag(
            Tag(
                tag_name='Journal',
                props=[
                    Prop[str](
                        prop_name='name',
                        data_type=PropType.STRING,
                        not_null=True,
                        comment='期刊名称',
                    ),
                    Prop[str](
                        prop_name='issn',
                        data_type=PropType.STRING,
                        comment='期刊ISSN编号',
                    ),
                ],
            ),
            check_exist=True,
        )

        store.create_edge_type(Edge(edge_name='AUTHORED'), check_exist=True)
        store.create_edge_type(Edge(edge_name='PUBLISH_ON'), check_exist=True)
        store.create_edge_type(Edge(edge_name='HAS_KEYWORD'), check_exist=True)
        store.create_edge_type(Edge(edge_name='CITES'), check_exist=True)


def insert_paper(pubmed_data: PubMedData):
    pubmed_dict = pubmed_data.model_dump()
    authors: list[dict] = pubmed_dict.pop('author')
    journal: dict[str, str] = pubmed_dict.pop('journal')
    keywords: list[str] = pubmed_dict.pop('keywords')
    references: list[str] = pubmed_dict.pop('references')

    with NebulaGraphStore(
        address=nebula_cfg.HOST,
        port=nebula_cfg.PORT,
        username=nebula_cfg.USERNAME,
        password=nebula_cfg.PASSWORD,
    ) as store:
        store.use_space('pubmed')

        result = store.insert_vertex('Paper', pubmed_dict, pubmed_data.pmid)
        assert result.is_succeeded(), result.error_msg()

        for index, author in enumerate(authors):
            result = store.insert_vertex('Author', author, author['name'])
            assert result.is_succeeded(), result.error_msg()
            result = store.insert_edge(
                'AUTHORED', author['name'], pubmed_data.pmid, rank=(index + 1)
            )
            assert result.is_succeeded(), result.error_msg()

        if journal:
            result = store.insert_vertex('Journal', journal, journal['name'])
            assert result.is_succeeded(), result.error_msg()
            result = store.insert_edge('PUBLISH_ON', pubmed_data.pmid, journal['name'])
            assert result.is_succeeded(), result.error_msg()

        for keyword in keywords:
            result = store.insert_vertex('Keyword', {'text': keyword}, keyword)
            assert result.is_succeeded(), result.error_msg()
            result = store.insert_edge('HAS_KEYWORD', pubmed_data.pmid, keyword)
            assert result.is_succeeded(), result.error_msg()

        for reference in references:
            result = store.insert_edge('CITES', pubmed_data.pmid, reference)
            assert result.is_succeeded(), result.error_msg()


if __name__ == '__main__':
    init_pubmed_graph(True)
