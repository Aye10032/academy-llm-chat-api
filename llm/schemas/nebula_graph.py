from enum import StrEnum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T')


class VidType(StrEnum):
    STRING255 = 'FIXED_STRING(255)'
    INT64 = 'INT[64]'


class PropType(StrEnum):
    INT64 = 'int64'
    INT32 = 'int32'
    INT16 = 'int16'
    INT8 = 'int8'
    FLOAT = 'float'
    DOUBLE = 'double'
    BOOL = 'bool'
    STRING = 'string'
    DATE = 'date'
    TIME = 'time'
    DATETIME = 'datetime'
    TIMESTAMP = 'timestamp'
    DURATION = 'duration'
    GEO = 'geography'
    POINT = 'geography(point)'
    LINESTRING = 'geography(linestring)'
    POLYGON = 'geography(polygon)'


class Prop(BaseModel, Generic[T]):
    prop_name: str
    data_type: str
    not_null: bool = False
    default: Optional[T] = None
    comment: str = ''

    def to_ngql(self):
        prop_str = f'{self.prop_name} {self.data_type}'
        if self.not_null:
            prop_str += ' NOT NULL'

        if self.default is not None:
            if isinstance(self.default, str):
                prop_str += (
                    f' DEFAULT {self.default}'
                    if self.default.endswith('()')
                    else f' DEFAULT "{self.default}"'
                )
            else:
                prop_str += f' DEFAULT {self.default}'

        if self.comment:
            prop_str += f' COMMENT "{self.comment}"'

        return prop_str

    def __eq__(self, other):
        if isinstance(other, str):
            return self.prop_name == other
        return False


class NebulaBase(BaseModel):
    props: list[Prop] = Field(default_factory=list)
    ttl_duration: Optional[int] = None
    ttl_col: Optional[Prop] = None
    comment: str = ''


class Tag(NebulaBase):
    tag_name: str


class Edge(NebulaBase):
    edge_name: str