from pydantic import BaseModel, ValidationError, field_validator


class ArticleBlock(BaseModel):
    """用于存储文本的统一结构体

    Attributes:
        text: 此段落的文本
        text_level: 此段落的文本等级。0代表正文，1-6代表markdown格式下的1到6级标题
    """

    text: str
    text_level: int = 0

    @field_validator('text_level')
    def validate_text_level(cls, value):
        if not 0 <= value <= 6:
            raise ValidationError('text_level must be between 0 and 6')
        return value
