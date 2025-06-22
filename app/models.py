from typing import Optional
from sqlmodel import Field, SQLModel

class PersonBase(SQLModel):
    """ 人物基础模型，用于API输入输出 """
    chinese_name: str = Field(index=True)
    description: Optional[str] = None

class Person(PersonBase, table=True):
    """ 人物数据库表模型 """
    id: Optional[int] = Field(default=None, primary_key=True)
    photo_path: Optional[str] = None
    # 使用bytes存储numpy array转换后的二进制数据
    embedding: Optional[bytes] = None

# class PersonCreate(PersonBase):
#     """ 用于创建人物的输入模型 """
#     pass

class PersonRead(PersonBase):
    """ 用于读取人物的输出模型，包含ID和照片路径 """
    id: int
    photo_path: Optional[str]