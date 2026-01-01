import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
from ultralytics import YOLO
import pandas as pd
import zipfile
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.spatial.distance import euclidean

# Load YOLO model
def load_model():
    model_path = "project_files/best.pt"  # Update with your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    return model

# Detect potholes in a single image
def detect_potholes(image, model):
    image_copy = image.copy()
    results = model(image_copy)
    pothole_data = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence > 0.5:
                pothole_data.append([x1, y1, x2, y2, confidence])
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(image_copy, f"{confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image_copy, pothole_data

# Merge GPS with detections
def merge_gps_data(pothole_data, gps_data, frame_index):
    merged_data = []
    if not gps_data.empty and frame_index < len(gps_data):
        gps_lat, gps_lon = gps_data.iloc[frame_index][['Latitude', 'Longitude']]
        for x1, y1, x2, y2, confidence in pothole_data:
            merged_data.append([frame_index, gps_lat, gps_lon, x1, y1, x2, y2, confidence])
    return merged_data

# Helper to compute centroid
def compute_centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# Process video and extract pothole information
# def process_video(video_path, gps_data, model, temp_dir, progress_bar):
#     video = cv2.VideoCapture(video_path)
#     output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#     frames_folder = os.path.join(temp_dir, "frames")
#     os.makedirs(frames_folder, exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     pothole_results = []
#     gps_points_all = []
#     unique_potholes = []
#     unique_potholes_set = []  # (lat, lon) centroids for comparison
#     frame_index = 0

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break

#         detected_frame, pothole_data = detect_potholes(frame, model)
#         out.write(detected_frame)

#         if pothole_data:
#             frame_filename = f"frame_{frame_index:04d}.png"
#             cv2.imwrite(os.path.join(frames_folder, frame_filename), detected_frame)

#         merged = merge_gps_data(pothole_data, gps_data, frame_index)
#         pothole_results.extend(merged)

#         for item in merged:
#             lat, lon = item[1], item[2]
#             centroid = compute_centroid(item[3], item[4], item[5], item[6])
#             latlon_centroid = (lat, lon)

#             # Avoid double-counting similar potholes
#             is_unique = True
#             for existing in unique_potholes_set:
#                 if euclidean(latlon_centroid, existing) < 0.0002:
#                     is_unique = False
#                     break
#             if is_unique:
#                 unique_potholes.append(item)
#                 unique_potholes_set.append(latlon_centroid)
#                 gps_points_all.append((lat, lon))

#         frame_index += 1
#         progress_bar.progress(frame_index / total_frames)

#     video.release()
#     out.release()

#     pothole_df = pd.DataFrame(pothole_results,
#         columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
#     pothole_csv_path = os.path.join(temp_dir, "pothole_coordinates.csv")
#     pothole_df.to_csv(pothole_csv_path, index=False)

#     unique_df = pd.DataFrame(unique_potholes,
#         columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
#     unique_csv_path = os.path.join(temp_dir, "unique_potholes.csv")
#     unique_df.to_csv(unique_csv_path, index=False)

#     return output_video_path, frames_folder, pothole_csv_path, unique_csv_path, gps_points_all

def process_video(video_path, gps_data, model, temp_dir, progress_bar):
    video = cv2.VideoCapture(video_path)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    frames_folder = os.path.join(temp_dir, "frames")
    os.makedirs(frames_folder, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pothole_results = []
    gps_points_all = []
    unique_potholes = []
    unique_potholes_set = []  # (lat, lon) centroids for comparison
    frame_index = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        detected_frame, pothole_data = detect_potholes(frame, model)
        out.write(detected_frame)

        if pothole_data:
            frame_filename = f"frame_{frame_index:04d}.png"
            cv2.imwrite(os.path.join(frames_folder, frame_filename), detected_frame)

        merged = merge_gps_data(pothole_data, gps_data, frame_index)
        pothole_results.extend(merged)

        for item in merged:
            lat, lon = item[1], item[2]
            confidence = item[7]

            # Apply confidence threshold for uniqueness
            if confidence < 0.65:
                continue

            latlon_centroid = (lat, lon)

            # Avoid double-counting nearby potholes (threshold ~5 meters)
            is_unique = True
            for existing in unique_potholes_set:
                if euclidean(latlon_centroid, existing) < 0.00005:
                    is_unique = False
                    break
            if is_unique:
                unique_potholes.append(item)
                unique_potholes_set.append(latlon_centroid)
                gps_points_all.append((lat, lon))

        frame_index += 1
        progress_bar.progress(frame_index / total_frames)

    video.release()
    out.release()

    pothole_df = pd.DataFrame(pothole_results,
        columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
    pothole_csv_path = os.path.join(temp_dir, "pothole_coordinates.csv")
    pothole_df.to_csv(pothole_csv_path, index=False)

    unique_df = pd.DataFrame(unique_potholes,
        columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
    unique_csv_path = os.path.join(temp_dir, "unique_potholes.csv")
    unique_df.to_csv(unique_csv_path, index=False)

    return output_video_path, frames_folder, pothole_csv_path, unique_csv_path, gps_points_all


# Create map from GPS points
def create_pothole_map(gps_points, heatmap=False):
    if not gps_points:
        return None

    avg_lat = np.mean([lat for lat, _ in gps_points])
    avg_lon = np.mean([lon for _, lon in gps_points])
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16)

    if heatmap:
        HeatMap(gps_points, radius=15).add_to(m)
    else:
        for lat, lon in gps_points:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(m)

    return m

# Streamlit App Entry Point
def main():
    st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
    st.title("🛣️ YOLOv10n Pothole Detection System with GPS, Count, and Mapping")

    if "model" not in st.session_state:
        st.session_state.model = load_model()

    uploaded_file = st.file_uploader("Upload a video (Up to 1TB)...", type=["mp4", "avi", "mov"])
    uploaded_gps = st.file_uploader("Upload GPS coordinates (CSV file)...", type=["csv"])

    if uploaded_file and uploaded_gps:
        if st.button("Start Processing"):
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            gps_path = os.path.join(temp_dir, uploaded_gps.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            with open(gps_path, "wb") as f:
                f.write(uploaded_gps.read())

            gps_data = pd.read_csv(gps_path)

            st.subheader("Processing video...")
            progress_bar = st.progress(0)

            output_video_path, frames_folder, pothole_csv_path, unique_csv_path, gps_points_all = process_video(
                file_path, gps_data, st.session_state.model, temp_dir, progress_bar
            )

            zip_path = os.path.join(temp_dir, "processed_results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(output_video_path, "processed_video.mp4")
                zipf.write(pothole_csv_path, "pothole_coordinates.csv")
                zipf.write(unique_csv_path, "unique_potholes.csv")
                for frame in os.listdir(frames_folder):
                    zipf.write(os.path.join(frames_folder, frame), os.path.join("frames", frame))

            st.session_state.processed = {
                "output_video_path": output_video_path,
                "pothole_csv_path": pothole_csv_path,
                "unique_csv_path": unique_csv_path,
                "zip_path": zip_path,
                "gps_points_all": gps_points_all
            }
            st.session_state.download_clicked = False

    # After processing
    if "processed" in st.session_state and not st.session_state.get("download_clicked", False):
        st.success("✅ Processing complete!")

        st.subheader("🎥 Processed Video")
        with open(st.session_state.processed["output_video_path"], "rb") as video_file:
            st.video(video_file.read())

        gps_points_all = st.session_state.processed.get("gps_points_all", [])

        st.info(f"🕳️ **Unique Potholes Detected:** {len(gps_points_all)}")

        st.subheader("🗺️ Pothole Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Marker Map**")
            pothole_map = create_pothole_map(gps_points_all, heatmap=False)
            if pothole_map:
                st_folium(pothole_map, width=600, height=450)

        with col2:
            st.markdown("**🔥 Heatmap**")
            heat_map = create_pothole_map(gps_points_all, heatmap=True)
            if heat_map:
                st_folium(heat_map, width=600, height=450)

        st.subheader("📥 Download Results")
        with open(st.session_state.processed["zip_path"], "rb") as file:
            if st.download_button("Download All Processed Data (ZIP)", file, file_name="processed_results.zip", mime="application/zip"):
                st.session_state.download_clicked = True
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()

