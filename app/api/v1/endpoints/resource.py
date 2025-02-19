import os
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

router = APIRouter()


@router.get(
    '/pdf/{file_path:path}',
    description='获取PDF文件内容',
)
async def get_pdf_file(file_path: str):
    """
    获取PDF文件内容

    Args:
        file_path: PDF文件相对路径
    """

    file_path = file_path.lstrip('./').lstrip('/').strip()
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='文件不存在')

    if not file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='仅支持PDF文件')

    filename = os.path.basename(file_path)
    encoded_filename = quote(filename)

    return FileResponse(
        file_path,
        media_type='application/pdf',
        headers={
            'Content-Disposition': f'inline; filename="{encoded_filename}"',
            'Access-Control-Allow-Origin': '*',
        },
    )
