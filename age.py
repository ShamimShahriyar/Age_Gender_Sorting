import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from shutil import copy2
from tqdm import tqdm
from deepface import DeepFace

# Load Haarcascade Face Detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Define Age Groups for Sorting
AGE_GROUPS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50),
              (51, 60), (61, 70), (71, 80), (81, 95)]

# Tkinter Window
root = tk.Tk()
root.title("Face Sorting by Age")
root.geometry("600x400")

input_folder = tk.StringVar()
output_folder = tk.StringVar()

def select_input_folder():
    folder = filedialog.askdirectory()
    input_folder.set(folder)

def select_output_folder():
    folder = filedialog.askdirectory()
    output_folder.set(folder)

def classify_age(face_img):
    """Predict age using DeepFace."""
    try:
        result = DeepFace.analyze(face_img, actions=["age"], enforce_detection=False)
        age = result[0]['age']
        
        # Ensure age is within a valid range
        age = max(0, min(age, 100))

        print(f"✔️ Predicted Age: {age}")  
        return age
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        return None

def get_age_group(age):
    """Determine the correct age group for sorting."""
    for start, end in AGE_GROUPS:
        if start <= age <= end:
            return f"{start}-{end}"
    return "unknown"

def process_images():
    input_path = input_folder.get()
    output_path = output_folder.get()

    if not input_path or not output_path:
        messagebox.showwarning("Warning", "Please select input and output folders!")
        return

    # Create Age-Group Folders
    for start, end in AGE_GROUPS:
        os.makedirs(os.path.join(output_path, f"{start}-{end}"), exist_ok=True)

    log_text.insert(tk.END, f"Processing images in: {input_path}\n")
    root.update()

    for img_name in tqdm(os.listdir(input_path)):
        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            log_text.insert(tk.END, f"Skipping {img_name}: Unable to read image.\n")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            log_text.insert(tk.END, f"No face detected in {img_name}. Skipping.\n")
            continue

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]

            if face.size == 0:
                continue

            age = classify_age(face)
            if age is None:
                log_text.insert(tk.END, f"Error predicting age for {img_name}. Skipping.\n")
                continue

            age_group = get_age_group(age)

            # DEBUGGING: Print detected age group assignment
            print(f"Image: {img_name} -> Age: {age}, Group: {age_group}")

            # Save face in the corresponding folder
            dest_path = os.path.join(output_path, age_group, f"{age}_{img_name}")
            copy2(img_path, dest_path)

            log_text.insert(tk.END, f"{img_name} -> Age: {age}, Group: {age_group}\n")
            root.update()

    messagebox.showinfo("Completed", "Image processing completed!")

# UI Layout
tk.Label(root, text="Input Folder:").pack()
tk.Entry(root, textvariable=input_folder, width=50).pack()
tk.Button(root, text="Browse", command=select_input_folder).pack()

tk.Label(root, text="Output Folder:").pack()
tk.Entry(root, textvariable=output_folder, width=50).pack()
tk.Button(root, text="Browse", command=select_output_folder).pack()

tk.Button(root, text="Start Processing", command=process_images, bg="green", fg="white").pack(pady=10)

log_text = scrolledtext.ScrolledText(root, width=70, height=10)
log_text.pack()

root.mainloop()