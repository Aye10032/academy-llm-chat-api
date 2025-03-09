from datetime import date

from loguru import logger

from llm.rag.graph_storage import NebulaGraphStore
from llm.schemas.nebula_graph import PropType, Prop, Tag, Edge, VidType


def init_pubmed_graph(drop_old: bool = False):
    with NebulaGraphStore(address='172.18.19.150') as store:
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
                        not_null=True,
                        comment='期刊ISSN编号',
                    ),
                ],
            ),
            check_exist=True,
        )

        store.create_edge(
            Edge(
                edge_name='AUTHORED',
                props=[Prop[int](prop_name='author_order', data_type=PropType.INT8)],
            ),
            check_exist=True,
        )
        store.create_edge(Edge(edge_name='PUBLISH_ON'), check_exist=True)
        store.create_edge(Edge(edge_name='HAS_KEYWORD'), check_exist=True)
        store.create_edge(Edge(edge_name='CITES'), check_exist=True)

def insert_paper():
    ...


if __name__ == '__main__':
    init_pubmed_graph(True)