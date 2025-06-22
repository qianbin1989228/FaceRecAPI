from sqlmodel import Session, select
from app import models

def get_person(db: Session, person_id: int):
    '''
    根据ID获取人物信息
    :param db: 数据库会话
    :param person_id: 人物ID
    :return: 人物信息
    '''
    return db.get(models.Person, person_id)

def get_persons(db: Session, skip: int = 0, limit: int = 100):
    '''
    获取所有人物信息
    :param db: 数据库会话
    :param skip: 跳过的记录数
    :param limit: 限制返回的记录数
    :return: 人物信息列表
    '''
    return db.exec(select(models.Person).offset(skip).limit(limit)).all()

def create_person(db: Session, person: models.Person):
    '''
    创建人物信息
    :param db: 数据库会话
    :param person: 人物信息
    :return: 创建后的人物信息
    '''
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

def delete_person(db: Session, person_id: int):
    '''
    删除人物信息
    :param db: 数据库会话
    :param person_id: 人物ID
    :return: 删除的人物信息
    '''
    person = db.get(models.Person, person_id)
    if not person:
        return None
    db.delete(person)
    db.commit()
    return person