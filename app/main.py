import cv2
import numpy as np
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from app import crud, models, face_utils
from app.database import get_session, create_db_and_tables

# --- FastAPI 应用初始化 ---
app = FastAPI(title="人脸识别API系统")

# 挂载静态文件目录，用于存放上传的图片
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. 挂载 static 文件夹，前端可以通过 /static/... 访问
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
# 2. 挂载 media 文件夹，用于访问上传的图片
app.mount("/media", StaticFiles(directory=BASE_DIR / "media"), name="media")
# 3. 配置 Jinja2 模板引擎
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- 应用启动时创建数据库表 ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- 人物管理API ---
@app.post("/persons/", response_model=models.PersonRead)
def create_person_api(
    chinese_name: str = Form(...),
    description: str = Form(None),
    photo: UploadFile = File(...),
    db: Session = Depends(get_session)
):
    # 1. 读取上传的图片
    contents = photo.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="照片读取异常，请检查源文件。")

    # 2. 人脸检测
    face_image = face_utils.detect_and_extract_face(img)
    if face_image is None:
        raise HTTPException(status_code=400, detail="未检测到人脸，请上传有效照片。")
    
    # 3. 提取特征向量
    embedding = face_utils.get_embedding(face_image)

    # 4. 保存裁剪后的人脸图片
    media_dir = BASE_DIR / "media" / "person_photos"
    media_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = media_dir / filename
    cv2.imwrite(str(save_path), face_image)

    # 5. 创建数据库记录
    person_data = models.Person(
        chinese_name=chinese_name,
        description=description,
        photo_path=f"/media/person_photos/{filename}",
        embedding=embedding.tobytes() # 将numpy array转为bytes存储
    )
    
    return crud.create_person(db, person_data)

@app.get("/persons/", response_model=List[models.PersonRead])
def read_persons_api(skip: int = 0, limit: int = 100, db: Session = Depends(get_session)):
    persons = crud.get_persons(db, skip=skip, limit=limit)
    return persons

@app.delete("/persons/{person_id}")
def delete_person_api(person_id: int, db: Session = Depends(get_session)):
    deleted_person = crud.delete_person(db, person_id=person_id)
    if not deleted_person:
        raise HTTPException(status_code=404, detail="人物不存在")
    return {"message": f"人物 {deleted_person.chinese_name} 已删除"}


# --- 在线人脸识别API ---
@app.post("/recognize/")
def recognize_face_api(photo: UploadFile = File(...), db: Session = Depends(get_session)):
    # 1. 读取和处理上传的图片
    contents = photo.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="照片读取异常，请检查源文件。")

    face_image = face_utils.detect_and_extract_face(img)
    if face_image is None:
        return JSONResponse(status_code=400, content={"message": "未检测到人脸"})
    
    # 2. 提取待识别图片的特征
    query_embedding = face_utils.get_embedding(face_image)
    
    # 3. 从数据库获取所有人脸特征
    all_persons = crud.get_persons(db, limit=1000) # 假设人物库小于1000
    if not all_persons:
        return JSONResponse(status_code=404, content={"message": "人物库为空"})
    
    db_embeddings = [np.frombuffer(p.embedding, dtype=np.float32) for p in all_persons]
    
    # 4. 计算余弦相似度
    similarities = [np.dot(query_embedding, db_emb) for db_emb in db_embeddings]
    
    # 5. 找到最相似的人
    best_match_index = np.argmax(similarities)
    max_similarity = similarities[best_match_index]
    
    # 设定一个阈值，低于该阈值则认为不是同一个人
    if max_similarity < 0.5:
        return {"name": "未找到匹配人物", "similarity": f"{max_similarity*100:.2f}%"}
        
    most_similar_person = all_persons[best_match_index]
    
    result = {
        'name': most_similar_person.chinese_name,
        'similarity': f"{max_similarity * 100:.2f}%",
        'photo_url': most_similar_person.photo_path
    }
    return result


 # --- 新增：服务前端页面的根路由 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 返回 index.html 模板
    return templates.TemplateResponse("index.html", {"request": request})