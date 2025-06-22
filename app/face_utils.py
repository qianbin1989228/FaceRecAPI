from typing import Optional
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
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) # 去除颜色
    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
    result = embedding_model.predict(face_image)
    embedding = result.embedding
    embedding = np.array(embedding, dtype=np.float32)  # 使用float32以减小存储
    # L2 归一化
    embedding /= np.linalg.norm(embedding)
    return embedding