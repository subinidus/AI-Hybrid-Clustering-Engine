# 🖼️ AI Hybrid Image Clustering Engine
> **Super Resolution + Object Detection + Semantic Embedding**을 결합한 고성능 이미지 클러스터링 파이프라인

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLOv10](https://img.shields.io/badge/YOLO-v10s-green)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-black)

## 📌 Introduction
이 프로젝트는 대량의 비정형 이미지 데이터를 **의미론적(Semantic)으로 군집화**하고, **3D 공간에 시각화**하기 위한 데이터 처리 엔진입니다.
단순히 이미지 전체 특징을 추출하는 기존 방식과 달리, 저화질 개선(SR) → 객체 탐지(YOLO) → 특징 추출(CLIP)의 3단계 하이브리드 파이프라인을 구축하여 클러스터링의 정확도를 극대화했습니다.

## 🛠️ Architecture Pipeline
이 엔진은 다음 4단계 과정을 통해 데이터를 처리합니다:

1.  **Adaptive Super Resolution (EDSR)**: 128px 미만의 저해상도 이미지를 자동으로 감지하여 3배 업스케일링(Upscaling) 수행.
2.  **Object-Centric Cropping (YOLOv10s)**: 이미지 내의 불필요한 배경(Background Noise)을 제거하고, 핵심 객체(Main Object)만 크롭하여 특징 추출의 정확도 향상.
3.  **Semantic Embedding (CLIP ViT-B/32)**: 텍스트-이미지 간의 의미적 연관성을 학습한 CLIP 모델을 사용하여 512차원 고수준 특징 벡터 추출.
4.  **Auto-Tuning Clustering (UMAP + DBSCAN)**:
    * **PCA & UMAP**: 3차원으로 차원 축소 (시각화 용이성 및 노이즈 제거).
    * **Knee Locator**: DBSCAN의 최적 `epsilon` 값을 수학적으로 자동 계산하여 하이퍼파라미터 튜닝 자동화.

## ✨ Key Features
* **🧩 Multi-Model Ensemble**: EDSR, YOLOv10, CLIP 등 SOTA(State-of-the-Art) 모델들의 장점을 결합.
* **🔍 Conditional Processing**: 모든 이미지를 업스케일링하지 않고, 작은 이미지에만 자원을 집중하여 처리 효율성 확보.
* **🤖 Automated Tuning**: 데이터 분포에 따라 클러스터링 밀도(Eps)를 동적으로 조절하는 알고리즘 탑재.
* **📊 Quality Assessment**: 클러스터 내의 데이터 응집도(Spread)를 계산하여 그룹핑 품질(High/Low) 자동 평가.

## 💻 Tech Stack
* **Core**: Python, PyTorch, OpenCV
* **Models**: YOLOv10 (Ultralytics), CLIP (OpenAI), EDSR (Super Resolution)
* **ML/Math**: Scikit-learn (PCA, DBSCAN), UMAP, Kneed (Elbow point detection)

## 🚀 Usage

### 1. Installation
```
pip install -r requirements.txt
```

### 2. Run Engine
```
python cluster_engine.py --image_dir ./data/images --output ./results/clustering_result.json
```
* image_dir: 클러스터링할 이미지가 담긴 폴더 경로
* output: 결과가 저장될 JSON 파일 경로

## 📂 Output Structure (JSON)
결과 파일은 3D 시각화 플랫폼(Three.js 등)에서 바로 사용할 수 있는 형태로 저장됩니다.

```
[
    {
        "filename": "data/images/player_01.jpg",
        "label": "player",
        "x": 3.421,
        "y": -1.205,
        "z": 5.112,
        "group": 1,
        "quality": "high"
    },
    ...
]
```

## 📜 License
MIT License
