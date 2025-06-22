from sqlmodel import create_engine, Session, SQLModel

# 使用SQLite数据库文件
DATABASE_URL = "sqlite:///database.db"
# connect_args 是SQLite特有的，确保单线程操作安全
engine = create_engine(DATABASE_URL, echo=True, connect_args={"check_same_thread": False})

def create_db_and_tables():
    """ 创建数据库和表 """
    SQLModel.metadata.create_all(engine)

def get_session():
    """ 依赖注入：获取数据库会话 """
    with Session(engine) as session:
        yield session