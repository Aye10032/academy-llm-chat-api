import time
from datetime import date, datetime
from typing import Any, Optional

from loguru import logger
from nebula3.Config import Config as NConfig
from nebula3.data.ResultSet import ResultSet
from nebula3.gclient.net import ConnectionPool

from llm.schemas.nebula_graph import Edge, PropType, Tag


class NebulaGraphStore:
    def __init__(
        self,
        address: str = '127.0.0.1',
        port: int = 9669,
        username: str = 'root',
        password: str = 'nebula',
    ):
        self.address = address
        self.port = port
        self.username = username
        self.password = password

        self.__init_connection()

    def __init_connection(self):
        n_config = NConfig()
        n_config.max_connection_pool_size = 2

        self.connection_pool = ConnectionPool()
        assert self.connection_pool.init([(self.address, self.port)], n_config)

        self.client = self.connection_pool.get_session(self.username, self.password)
        assert self.client is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.release()
        self.connection_pool.close()

    def create_space(
        self,
        space_name: str,
        partition_num: int = 20,
        replica_factor: int = 1,
        comment: str = None,
        *,
        vid_type: str,
        check_exist: bool = True,
    ) -> ResultSet:
        """创建图空间

        等效于以下SQL语句：
        CREATE SPACE [IF NOT EXISTS] <graph_space_name> (
        [partition_num = <partition_number>,]
        [replica_factor = <replica_number>,]
        vid_type = {FIXED_STRING(<N>) | INT[64]}
        )
        [COMMENT = '<comment>'];

        Args:
            space_name: 在NebulaGraph实例中唯一标识一个图空间，仅支持 1~4 字节的 UTF-8 编码字符，包括英文字母（区分大小写）、数字、中文等
            partition_num: 指定图空间的分片数量。建议设置为集群中硬盘数量的 20 倍（HDD 硬盘建议为 2 倍）
            replica_factor: 指定每个分片的副本数量。建议在生产环境中设置为 3，在测试环境中设置为 1。副本数量必须是奇数
            comment: 图空间的描述
            vid_type: 指定点 ID 的数据类型
            check_exist:

        Returns:
            操作结果
        """

        assert vid_type == 'INT[64]' or vid_type.startswith('FIXED_STRING')

        if check_exist:
            stmt = (
                f'CREATE SPACE IF NOT EXISTS {space_name} ('
                f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type})'
            )
        else:
            stmt = (
                f'CREATE SPACE {space_name} ('
                f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type})'
            )

        if comment is not None:
            stmt += f' comment="{comment}"'

        result = self.client.execute(stmt)

        logger.info('Creating graph database, please wait...')
        time.sleep(20)
        logger.info('done')
        return result

    def use_space(self, space_name: str) -> ResultSet:
        """切换到指定图空间

        等效于以下SQL语句：
        USE <graph_space_name>;

        Args:
            space_name: 图空间唯一标识

        Returns:
            操作结果
        """
        result = self.client.execute(f'USE {space_name};')
        return result

    def clear_space(self, space_name: str, *, check_exist: bool = True) -> ResultSet:
        """清空图空间中的点和边，但不会删除图空间本身以及其中的 Schema 信息

        等效于以下SQL语句：
        CLEAR SPACE [IF EXISTS] <graph_space_name>

        Args:
            space_name: 图空间唯一标识
            check_exist:

        Returns:
            操作结果
        """
        stmt = (
            f'CLEAR SPACE IF EXISTS {space_name};' if check_exist else f'CLEAR SPACE {space_name};'
        )

        result = self.client.execute(stmt)
        return result

    def drop_space(self, space_name: str, *, check_exist: bool = True) -> ResultSet:
        """删除指定图空间

        等效于以下SQL语句：
        DROP SPACE [IF EXISTS] <graph_space_name>

        Args:
            space_name: 图空间唯一标识
            check_exist:

        Returns:
            操作结果
        """
        stmt = f'DROP SPACE IF EXISTS {space_name};' if check_exist else f'DROP SPACE {space_name};'
        result = self.client.execute(stmt)

        logger.info('Deleting graph database, please wait...')
        time.sleep(5)
        logger.info('done')
        return result

    def create_tag(
        self,
        tag: Tag,
        *,
        check_exist: bool = True,
    ) -> ResultSet:
        """创建TAG

        等效于以下SQL语句：
        CREATE TAG [IF NOT EXISTS] <tag_name> (
        <prop_name> <data_type>
        [NULL | NOT NULL]
        [DEFAULT <default_value>]
        [COMMENT '<comment>']
        [{, <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']} ...]
        )
        [TTL_DURATION = <ttl_duration>]
        [TTL_COL = <prop_name>]
        [COMMENT = '<comment>']

        Args:
            tag: Tag对象
            check_exist:

        Returns:
            操作结果
        """

        prop_str = ' ,'.join([prop.to_ngql() for prop in tag.props])
        stmt_1 = (
            f'CREATE TAG IF NOT EXISTS {tag.tag_name}({prop_str})'
            if check_exist
            else f'CREATE TAG {tag.tag_name}({prop_str})'
        )
        stmt_2 = []
        if tag.ttl_duration is not None:
            assert tag.ttl_col and tag.ttl_col.data_type in [
                PropType.INT64,
                PropType.INT32,
                PropType.INT16,
                PropType.INT8,
                PropType.TIMESTAMP,
            ]
            stmt_2.append(f'TTL_DURATION = {tag.ttl_duration}')
            stmt_2.append(f'TTL_COL = "{tag.ttl_col.prop_name}"')

        if tag.comment:
            stmt_2.append(f'COMMENT = "{tag.comment}"')

        stmt = f'{stmt_1} {", ".join(stmt_2)};'
        result = self.client.execute(stmt)

        return result

    def drop_tag(self, tag_name: str, *, check_exist: bool = True) -> ResultSet:
        """删除当前工作空间内所有点上的指定 Tag

        等效于以下SQL语句：
        DROP TAG [IF EXISTS] <tag_name>;

        Args:
            tag_name: 要删除的tag名称
            check_exist:

        Returns:
            操作结果
        """

        stmt = f'DROP TAG IF EXISTS {tag_name};' if check_exist else f'DROP TAG {tag_name};'
        result = self.client.execute(stmt)
        return result

    def delete_tag(self, tag_names: list[str], vid_list: list[str]) -> ResultSet:
        """删除特定TAG

        等效于以下SQL语句：
        DELETE TAG <tag_name_list> FROM <VID_list>;

        Args:
            tag_names: 要删除的tag名称
            vid_list: 要删除的tag vid列表

        Returns:
            操作结果
        """

        if len(tag_names) == 0:
            tag_str = '*'
        else:
            tag_str = ','.join(tag_names)

        vid_list = [f'"{vid}"' for vid in vid_list]

        stmt = f'DELETE TAG {tag_str} FROM {",".join(vid_list)};'
        result = self.client.execute(stmt)
        return result

    def create_edge_type(
        self,
        edge: Edge,
        *,
        check_exist: bool = True,
    ) -> ResultSet:
        """创建边类型

        等效于以下SQL语句：
        CREATE EDGE [IF NOT EXISTS] <edge_type_name>(
          <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']
          [{, <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']} ...]
        )
        [TTL_DURATION = <ttl_duration>]
        [TTL_COL = <prop_name>]
        [COMMENT = '<comment>'];

        Args:
            edge: 边对象
            check_exist:

        Returns:
            操作结果
        """

        prop_str = ' ,'.join([prop.to_ngql() for prop in edge.props])
        stmt_1 = (
            f'CREATE EDGE IF NOT EXISTS {edge.edge_name}({prop_str})'
            if check_exist
            else f'CREATE EDGE {edge.edge_name}({prop_str})'
        )
        stmt_2 = []
        if edge.ttl_duration is not None:
            assert edge.ttl_col and edge.ttl_col.data_type in [
                PropType.INT64,
                PropType.INT32,
                PropType.INT16,
                PropType.INT8,
                PropType.TIMESTAMP,
            ]
            stmt_2.append(f'TTL_DURATION = {edge.ttl_duration}')
            stmt_2.append(f'TTL_COL = "{edge.ttl_col.prop_name}"')

        if edge.comment:
            stmt_2.append(f'COMMENT = "{edge.comment}"')

        stmt = f'{stmt_1} {", ".join(stmt_2)};'
        result = self.client.execute(stmt)
        return result

    def drop_edge_type(self, edge_name: str, *, check_exist: bool = True) -> ResultSet:
        """删除当前工作空间内的指定 Edge type

        DROP EDGE [IF EXISTS] <edge_type_name>;

        Args:
            edge_name: edge_name: 要删除的edge type名称
            check_exist:

        Returns:
            操作结果
        """

        stmt = f'DROP EDGE IF EXISTS {edge_name};' if check_exist else f'DROP EDGE {edge_name};'
        result = self.client.execute(stmt)
        return result

    def insert_vertex(
        self,
        tag_name: str,
        props: dict[str, Any],
        vid: str | int,
        *,
        check_exist: bool = True,
    ):
        """插入点

        INSERT VERTEX [IF NOT EXISTS] <tag_name> ([prop_name_list]) VALUES <vid>: ([prop_value_list])

        Args:
            tag_name: 节点类型名称
            props: 节点属性值
            vid: 节点索引名称
            check_exist:

        Returns:
            操作结果
        """
        prop_names = props.keys()
        prop_values = props.values()

        stmt_list = ['INSERT VERTEX']

        if check_exist:
            stmt_list.append('IF NOT EXISTS')

        stmt_list.extend([tag_name, f'({", ".join(prop_names)})'])

        if isinstance(vid, str):
            stmt_list.extend(['VALUES', f'"{vid}":'])
        else:
            stmt_list.extend(['VALUES', f'{vid}:'])

        prop_values_strs = []
        for prop_value in prop_values:
            if isinstance(prop_value, str):
                prop_values_strs.append(f'"{prop_value}"')
            elif isinstance(prop_value, date):
                prop_values_strs.append(f'date("{prop_value}")')
            elif isinstance(prop_value, datetime):
                prop_values_strs.append(f'datetime("{prop_value}")')
            else:
                prop_values_strs.append(f'{prop_value}')

        stmt_list.append(f'({", ".join(prop_values_strs)});')

        stmt = ' '.join(stmt_list)
        result = self.client.execute(stmt)
        return result

    def insert_edge(
        self,
        edge_type: str,
        src_vid: str | int,
        dst_vid: str | int,
        props: Optional[dict[str, Any]] = None,
        rank: int = 0,
        *,
        check_exist: bool = True,
    ):
        """插入边
        允许悬挂边（Dangling edge）。因此可以在起点或者终点存在前先写边

        INSERT EDGE [IF NOT EXISTS] <edge_type> ( <prop_name_list> ) VALUES <src_vid> -> <dst_vid>[@<rank>] : ( <prop_value_list> );
        Args:
            edge_type: 边关联的 Edge type
            props: 属性列表
            src_vid: 起始点 ID
            dst_vid: 目的点 ID
            rank: 边的 rank 值
            check_exist:

        Returns:
            操作结果
        """

        if props is None:
            props = {}

        prop_names = props.keys()
        prop_values = props.values()

        stmt_list = ['INSERT EDGE']

        if check_exist:
            stmt_list.append('IF NOT EXISTS')

        stmt_list.extend([edge_type, f'({", ".join(prop_names)})'])

        src_vid_str = f'"{src_vid}"' if isinstance(src_vid, str) else f'{src_vid}'
        dst_vid_str = f'"{dst_vid}"' if isinstance(dst_vid, str) else f'{dst_vid}'

        if rank:
            stmt_list.append(f'VALUES {src_vid_str}->{dst_vid_str}@{rank}:')
        else:
            stmt_list.append(f'VALUES {src_vid_str}->{dst_vid_str}:')

        prop_values_strs = []
        for prop_value in prop_values:
            if isinstance(prop_value, str):
                prop_values_strs.append(f'"{prop_value}"')
            elif isinstance(prop_value, date):
                prop_values_strs.append(f'date("{prop_value}")')
            elif isinstance(prop_value, datetime):
                prop_values_strs.append(f'datetime("{prop_value}")')
            else:
                prop_values_strs.append(f'{prop_value}')

        stmt_list.append(f'({", ".join(prop_values_strs)});')

        stmt = ' '.join(stmt_list)
        logger.debug(stmt)
        result = self.client.execute(stmt)
        return result