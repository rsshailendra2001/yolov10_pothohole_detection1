# 🛣️ Pothole Detection using YOLOv10n, YOLO11n, and YOLO11s

This repository contains the implementation of our study on **pothole detection and mapping** using recent YOLO models (**YOLOv10n, YOLO11n, YOLO11s**).  
The work was conducted to provide a **lightweight, real-time, and cost-effective framework** for infrastructure monitoring.

---

## 📌 Features
- Implementation of **YOLOv10n, YOLO11n, and YOLO11s** for pothole detection.  
- Pre-trained **weights, logs, and detection outputs** for all three models are included in this repository.  
- Interactive **Streamlit app (`app.py`)** for real-time detection and visualization.  
- Users can easily switch between YOLOv10n, YOLO11n, and YOLO11s by updating the weight path.  
- Integration with **GPS and GIS mapping** for geographic localization (as described in the manuscript).  
- Repository designed for **reproducibility** of results.  

---

## 📂 Repository Structure
```

├── YOLOv10n(325epoch)/ # Trained weights, logs, detection results for YOLOv10n
├── YOLO11n(325epoch)/ # Trained weights, logs, detection results for YOLO11n
├── YOLO11s(325epoch)/ # Trained weights, logs, detection results for YOLO11s
├── project_files/ # Extra utility files
├── utils/ # Helper functions
├── .streamlit/ # Streamlit configuration
├── app.py # Streamlit app
├── YOLO_run.ipynb # Jupyter Notebook (training & evaluation workflow)
├── requirements.txt # Python dependencies
├── packages.txt
└── README.md # Documentation

````

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/Abhishek7281/YOLO-Pothole-Mapping.git
cd YOLO-Pothole-Mapping
````

 The repository already contains the trained results and weights for all three models, each stored in a dedicated folder:

 YOLOv10n(325epoch) → includes trained weights, logs, and detection outputs **for** YOLOv10n

 YOLO11n(325epoch) → includes trained weights, logs, and detection outputs for YOLO11n

 YOLO11s(325epoch) → includes trained weights, logs, and detection outputs for YOLO11s

 The repository already contains the trained weights, logs, and outputs for all three models:

YOLOv10n(325epoch) → weights, logs, detection outputs for YOLOv10n

YOLO11n(325epoch) → weights, logs, detection outputs for YOLO11n

YOLO11s(325epoch) → weights, logs, detection outputs for YOLO11s

👉 Users can directly explore these folders to reproduce the results reported in the manuscript without retraining.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Streamlit app:

```bash
streamlit run app.py
```

* By default, the app runs with **YOLOv10n** weights.
* To use **YOLO11n** or **YOLO11s**, replace the weight file in `project_files/` and update the path in `app.py`.

Example (inside `app.py`):

```python
# Change this line to point to your desired YOLO weights
weights_path = "project_files/yolov11n.pt"
```

---

### 🔹 Access the hosted Streamlit app  

👉 [Live Streamlit Demo](https://yolov10pothoholedetection1-gy4agffbzk76s8rszcu3jy.streamlit.app/)  


## 📊 Results

* Results for **YOLOv10n, YOLO11n, and YOLO11s** are available in this repository.
* YOLO11n achieved the **best trade-off** between accuracy, inference time (1.1 ms), and model size (5.6 MB).

---

## 📁 Dataset

* The dataset used in this study is publicly available at the following link:
Google Drive Dataset Link : https://drive.google.com/file/d/1qEBV9wqBzuLbUBFnV_GBROCJLSoygrUT/view

This dataset was prepared using the RoboFlow platform and contains 13,767 images with corresponding annotations for pothole detection. It can be directly used for training and evaluating YOLOv10n, YOLO11n, and YOLO11s models.

---

## 🙏 Acknowledgement

We thank the reviewers for their valuable suggestions to make this repository open and reproducible for the research community.
