好的，遵照您的要求，我将参考您提供的 Django 项目文章，并将其核心逻辑与功能迁移到 FastAPI 框架上，使用 SQLModel 进行数据库操作，打造一篇关于《FastAPI+深度学习：打造商业级人脸识别系统》的博客。

---

## FastAPI+深度学习：打造商业级人脸识别系统



### 一、前言

在当今的AI浪潮中，人脸识别技术已渗透到我们生活的方方面面。构建一个高效、可扩展的人脸识别服务是许多开发者和企业的需求。传统的Django框架虽然功能强大，但对于追求极致性能和现代API开发的场景，[**FastAPI**](https://fastapi.tiangolo.com/) 无疑是更优的选择。

**为什么选择 FastAPI？**
*   **极致性能**：基于 Starlette 和 Pydantic，FastAPI 的性能与 NodeJS 和 Go 不相上下，是最高性能的 Python Web 框架之一。
*   **开发高效**：得益于 Python 的类型提示，代码自动补全、类型检查和错误排查能力大大增强，开发效率提升约 200%-300%。
*   **现代易用**：自动生成交互式 API 文档 (Swagger UI / ReDoc)，支持异步 `async/await`，并采用依赖注入系统，代码结构清晰，易于维护。

本文将带领您，利用 **FastAPI** 作为后端服务框架，**SQLModel** 作为现代化的数据库ORM，并结合强大的 **Dlib** 和 **ArcFace (FastDeploy)** 深度学习模型，从零开始构建一个结构简洁、功能完整、具备商业级潜力的人脸识别系统。


### 二、项目搭建

#### 2.1 环境安装

首先，确保您的系统中已安装 Python 3.8 或更高版本。然后，安装所有必要的库：

```bash
# Web框架和服务器
pip install fastapi uvicorn "python-multipart" jinja2

# 数据库ORM
pip install sqlmodel

# 深度学习与图像处理
pip install numpy opencv-python dlib fastdeploy-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```
**注意**：`dlib` 库的安装需要编译，可能会花费几分钟时间，请耐心等待。

#### 2.2 核心模型下载

与原项目一样，我们需要下载人脸检测和特征提取的模型。

1.  **Dlib 人脸关键点检测模型**：
    下载地址：[http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2.  **ArcFace 人脸特征提取模型**：
    ```bash
    wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx
    ```

下载并解压后，将 `shape_predictor_68_face_landmarks.dat` 和 `ms1mv3_arcface_r100.onnx` 这两个文件放在项目根目录下。

#### 2.3 项目结构

一个清晰的项目结构是高效开发的基石。我们采用如下简洁明了的结构：

```
/FaceRecAPI
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI应用主文件，包含API路由
│   ├── database.py      # 数据库连接与会话管理
│   ├── models.py        # SQLModel数据模型定义
│   ├── crud.py          # 数据库增删改查(CRUD)操作
│   └── face_utils.py    # 封装人脸检测与特征提取的核心函数
├── static/              # 存放CSS, JS等静态文件
│   └── ...
├── templates/           # 存放HTML模板文件
│   └── ...
├── media/               # 存放上传的人物照片
│   └── person_photos/
├── shape_predictor_68_face_landmarks.dat  # Dlib模型
└── ms1mv3_arcface_r100.onnx               # ArcFace模型
```

### 三、后端核心功能开发

#### 3.1 数据库与模型定义 (app/models.py)

**SQLModel** 结合了 Pydantic 和 SQLAlchemy 的优点，让我们能用一个类同时定义数据模型、API数据结构和数据库表结构。

```python
# app/models.py
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

class PersonCreate(PersonBase):
    """ 用于创建人物的输入模型 """
    pass

class PersonRead(PersonBase):
    """ 用于读取人物的输出模型，包含ID和照片路径 """
    id: int
    photo_path: Optional[str]
```

#### 3.2 数据库配置 (app/database.py)

我们使用轻量级的 SQLite 数据库进行演示，并配置数据库引擎和会话。

```python
# app/database.py
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
```

#### 3.3 封装人脸处理工具 (app/face_utils.py)

将所有与深度学习模型相关的代码（人脸检测、特征提取）封装到一个单独的文件中，使主逻辑更清晰。

```python
# app/face_utils.py
import dlib
import cv2
import numpy as np
import fastdeploy as fd
from pathlib import Path

# --- 模型初始化 ---
# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Dlib 人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(BASE_DIR / 'shape_predictor_68_face_landmarks.dat'))

# 2. ArcFace 特征提取模型
option = fd.RuntimeOption()
option.use_cpu()
embedding_model = fd.vision.faceid.ArcFace(str(BASE_DIR / 'ms1mv3_arcface_r100.onnx'), runtime_option=option)

def detect_and_extract_face(image: np.ndarray) -> Optional[np.ndarray]:
    """
    从图像中检测并裁剪出最大的人脸。
    :param image: OpenCV格式的图像 (BGR)
    :return: 裁剪后的人脸图像，如果未检测到则返回None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    
    # 找到最大的人脸
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    
    # 稍微扩大裁剪区域，以包含整个头部
    padding_h = int(0.4 * h)
    start_y = max(0, y - padding_h)
    
    face_image = image[start_y : y + h, x : x + w]
    return face_image

def get_embedding(face_image: np.ndarray) -> np.ndarray:
    """
    提取人脸图像的512维特征向量并进行归一化。
    :param face_image: 裁剪后的人脸图像
    :return: 归一化后的特征向量 (numpy array)
    """
    result = embedding_model.predict(face_image)
    embedding = result.embedding
    embedding = np.array(embedding, dtype=np.float32)  # 使用float32以减小存储
    # L2 归一化
    embedding /= np.linalg.norm(embedding)
    return embedding
```

#### 3.4 数据库操作 (app/crud.py)

创建专门的函数来处理与数据库的交互，实现关注点分离。

```python
# app/crud.py
from sqlmodel import Session, select
from . import models

def get_person(db: Session, person_id: int):
    return db.get(models.Person, person_id)

def get_persons(db: Session, skip: int = 0, limit: int = 100):
    return db.exec(select(models.Person).offset(skip).limit(limit)).all()

def create_person(db: Session, person: models.Person):
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

def delete_person(db: Session, person_id: int):
    person = db.get(models.Person, person_id)
    if not person:
        return None
    db.delete(person)
    db.commit()
    return person
```

#### 3.5 创建API端点 (app/main.py)

这是我们的主应用文件，它将所有部分串联起来，并定义API路由。

```python
# app/main.py
import cv2
import numpy as np
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from . import crud, models, face_utils
from .database import get_session, create_db_and_tables

# --- FastAPI 应用初始化 ---
app = FastAPI(title="人脸识别API系统")

# 挂载静态文件目录，用于存放上传的图片
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/media", StaticFiles(directory=BASE_DIR / "media"), name="media")

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
    # ... 省略删除文件的逻辑 ...
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

```

#### 3.6 启动应用

在项目根目录下，运行以下命令启动服务：

```bash
uvicorn app.main:app --reload
```

现在，打开浏览器访问 `http://127.0.0.1:8000/docs`，你将看到 FastAPI 自动生成的交互式 API 文档，可以在这里直接测试你的所有 API 接口！




### 四、前端界面与集成

虽然 FastAPI 专注于API，但我们也可以用它来服务一个简单的前端页面，以提供完整的用户体验。这部分可以大量复用原 Django 项目的 HTML 和 JS，只需做少量修改。

1.  **准备前端文件**：将原项目中的 `static` 和 `templates` 文件夹拷贝到新项目根目录。

2.  **修改`main.py`支持模板**：
    在 `app/main.py` 头部添加：
    ```python
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse
    from fastapi import Request

    # ... 其他 import ...

    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    # ... 在文件末尾添加前端路由 ...
    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("home.html", {"request": request})

    @app.get("/manage", response_class=HTMLResponse)
    async def read_manage_page(request: Request):
        return templates.TemplateResponse("manage.html", {"request": request})
    ```
3.  **修改HTML和JS**：
    *   将所有 Django 模板标签（如 `{% url '...' %}`, `{% static '...' %}`）替换为 FastAPI 的静态路径（如 `/manage`, `/static/css/bootstrap.min.css`）。
    *   修改 JavaScript 中的 `fetch` 请求，使其指向我们新创建的 FastAPI 端点（如 `/persons/`, `/recognize/`）。由于我们使用了 `Form` 数据，JS `fetch` 部分基本无需改动。

    例如，在线识别的 `home.html` 中的 JavaScript：
    ```javascript
    // ...
    function submitForm() {
        var form = document.getElementById('upload-form');
        var formData = new FormData(form);

        // API端点修改为 /recognize/
        fetch('/recognize/', {
            method: 'POST',
            body: formData,
            // FastAPI不需要CSRF Token
        })
       .then(response => response.json())
       .then(data => {
            // ... (处理返回结果的逻辑不变)
        })
       .catch(error => console.error('Error:', error));
    }
    // ...
    ```

### 五、优化建议

我们的系统已经可以良好地工作，但对于真正的商业级应用，还有一些优化方向：

1.  **向量检索数据库**：当人脸库达到数万甚至数百万级别时，线性遍历计算相似度会变得非常缓慢。可以引入 **FAISS** 或 **Milvus** 等专业的向量检索引擎。在系统启动时，将所有数据库中的人脸特征加载到向量库中，实现毫秒级的海量数据检索。

2.  **GPU 加速**：如果服务器配备了 GPU，应配置 FastDeploy 使用 GPU 进行推理。只需在初始化 `RuntimeOption` 时修改为 `option.use_gpu(0)`，即可将特征提取的速度提升数十倍。

3.  **异步任务处理**：对于添加人物这种包含文件读写、模型推理和数据库写入的耗时操作，可以利用 FastAPI 的 `BackgroundTasks` 将其放入后台执行。API 可以立即返回一个“处理中”的响应，提升用户体验。

### 六、总结

通过本文的引导，我们成功地使用 FastAPI、SQLModel 和深度学习模型构建了一个现代化、高性能的人脸识别系统。我们体验了 FastAPI 带来的开发效率和卓越性能，学习了如何通过 SQLModel 优雅地管理数据，并掌握了将深度学习模型集成到 Web 服务中的核心流程。

本项目不仅是一个功能完备的 Demo，更是一个可以轻松扩展和优化的起点。希望它能为您在探索 AI 应用开发的道路上提供有力的帮助。

**本项目完整代码链接**：[https://github.com/your-username/FaceRecAPI-Project](https://github.com/your-username/FaceRecAPI-Project) (请替换为您的实际GitHub链接)