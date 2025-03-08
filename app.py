import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "supersecretkey"

MODEL_PATH = "model/best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to('cpu')

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Function to process images and detect elephants
def process_images(upload_folder):
    results_data = []
    outliers_data = []
    outlier_threshold = 1.0

    for img_name in os.listdir(upload_folder):
        img_path = os.path.join(upload_folder, img_name)

        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading {img_name}, skipping...")
            continue

        # Run YOLO detection
        # results = model(img_path)  
        results = model(img_path, device='cpu')
        detections = len(results[0].boxes)
        results_data.append({"Image": img_name, "Detections": detections})

    df = pd.DataFrame(results_data)
    if df.empty:
        return None, None, None, None, False, []

    # Calculate Mean & Standard Deviation
    mean_count = df["Detections"].mean()
    std_dev = df["Detections"].std()

    if std_dev == 0:
        df["Outlier"] = "No"
    else:
        # Mark outliers
        df["Outlier"] = (df["Detections"] > (mean_count + outlier_threshold * std_dev))

    # Convert boolean to "Yes"/"No"
    df["Outlier"] = df["Outlier"].map({True: "Yes", False: "No"})

    # Save results to CSV
    results_csv_path = os.path.join(RESULTS_FOLDER, "elephant_detection_results.csv")
    df.to_csv(results_csv_path, index=False)

    # Save outliers separately
    outliers_df = df[df["Outlier"] == "Yes"]
    outliers_csv_path = os.path.join(RESULTS_FOLDER, "elephant_outliers.csv")
    outliers_df.to_csv(outliers_csv_path, index=False)

    # Disaster Prediction (Example)
    disaster_warning = False
    potential_disasters = []
    if len(outliers_df) > 4:
        disaster_warning = True
        potential_disasters = ["Poaching Alert", "Elephant Stampede", "Unusual Migration"]
        flash("🚨 Unusual elephant activity detected! Possible emergency!", "danger")

    return df, results_csv_path, outliers_df, outliers_csv_path, disaster_warning, potential_disasters


# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Clear previous uploads
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))

        # Save uploaded images
        uploaded_files = request.files.getlist("images")
        if not uploaded_files:
            flash("No files uploaded!", "danger")
            return redirect(url_for("index"))

        for file in uploaded_files:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        # Process images
        results_df, results_csv, outliers_df, outliers_csv, disaster_warning, potential_disasters = process_images(UPLOAD_FOLDER)

        if results_df is None:
            flash("No valid detections found!", "warning")
            return redirect(url_for("index"))

        return render_template("index.html", 
                               results=results_df.to_dict(orient="records"), 
                               csv_path=results_csv,
                               outliers_csv=outliers_csv,
                               disaster_warning=disaster_warning,
                               potential_disasters=potential_disasters)

    return render_template("index.html", results=None)


@app.route("/download_results")
def download_results_csv():
    return send_file(os.path.join(RESULTS_FOLDER, "ele_counts.csv"), as_attachment=True)

@app.route("/download_outliers")
def download_outliers_csv():
    return send_file(os.path.join(RESULTS_FOLDER, "ele_outliers.csv"), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
