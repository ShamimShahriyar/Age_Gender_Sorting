import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from shutil import copy2
from tqdm import tqdm
from deepface import DeepFace
import tensorflow as tf

# 🔥 Load Haarcascade Face Detector (Fix for NameError)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Tkinter Window
root = tk.Tk()
root.title("Face Sorting by Age & Gender")
root.geometry("750x550")

# Control Variables
input_folder = tk.StringVar()
output_folder = tk.StringVar()
batch_size = tk.IntVar(value=1)  # Default batch size = 1
gpu_enabled = tk.BooleanVar(value=True)  # Default to GPU enabled
gpu_memory_limit = tk.StringVar(value="2500")  # Default GPU memory limit = 2500MB
stop_processing = False  # Global flag to stop processing

def configure_gpu():
    """Set TensorFlow GPU/CPU settings based on user selection."""
    if not gpu_enabled.get():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("✔️ Running on CPU Only")
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                mem_limit = int(gpu_memory_limit.get())  # Convert input to integer
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
                    )
                print(f"✔️ GPU Enabled with Memory Growth (Limited to {mem_limit}MB)")
            except ValueError:
                print("⚠️ Invalid GPU memory limit! Using default 2500MB.")
                gpu_memory_limit.set("2500")  # Reset to default if invalid input
            except RuntimeError as e:
                print(f"⚠️ GPU Error: {e}")

# The rest of the script remains unchanged...

def select_input_folder():
    folder = filedialog.askdirectory()
    input_folder.set(folder)

def select_output_folder():
    folder = filedialog.askdirectory()
    output_folder.set(folder)

def classify_age_gender(face_img):
    """Predict age and gender using DeepFace (optimized for GPU)."""
    try:
        result = DeepFace.analyze(
            face_img, 
            actions=["age", "gender"], 
            enforce_detection=False, 
            detector_backend="retinaface"  # ✅ Use RetinaFace (better & optimized for GPU)
        )
        age = result[0]['age']
        gender = result[0]['dominant_gender'].capitalize()

        # Ensure age is within a valid range
        age = max(0, min(age, 100))

        print(f"✔️ Predicted Age: {age}, Gender: {gender}")  
        return age, gender
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        return None, None

def get_age_group(age):
    """Determine the correct age group for sorting."""
    AGE_GROUPS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50),
                  (51, 60), (61, 70), (71, 80), (81, 95)]
    for start, end in AGE_GROUPS:
        if start <= age <= end:
            return f"{start}-{end}"
    return "unknown"

def stop_process():
    """Stop the image processing loop."""
    global stop_processing
    stop_processing = True
    log_text.insert(tk.END, "⚠️ Processing Stopped by User!\n")
    root.update()

def process_images():
    global stop_processing
    stop_processing = False  # Reset stop flag

    input_path = input_folder.get()
    output_path = output_folder.get()

    if not input_path or not output_path:
        messagebox.showwarning("Warning", "Please select input and output folders!")
        return

    configure_gpu()  # Apply GPU/CPU settings before processing

    # Create Age-Group and Gender-Based Folders
    AGE_GROUPS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50),
                  (51, 60), (61, 70), (71, 80), (81, 95)]
    for start, end in AGE_GROUPS:
        for gender in ["Male", "Female"]:
            os.makedirs(os.path.join(output_path, f"{start}-{end}", gender), exist_ok=True)

    log_text.insert(tk.END, f"Processing images in: {input_path} (Batch Size: {batch_size.get()})\n")
    root.update()

    # 🔥 Fix: Process Only Image Files, Ignore Folders
    images = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    total_images = len(images)
    
    for i in tqdm(range(0, total_images, batch_size.get())):
        if stop_processing:
            log_text.insert(tk.END, "⚠️ Processing Stopped!\n")
            root.update()
            break

        batch = images[i:i + batch_size.get()]

        for img_name in batch:
            if stop_processing:
                log_text.insert(tk.END, "⚠️ Processing Stopped!\n")
                root.update()
                return

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

                age, gender = classify_age_gender(face)
                if age is None or gender is None:
                    log_text.insert(tk.END, f"Error predicting age/gender for {img_name}. Skipping.\n")
                    continue

                age_group = get_age_group(age)

                print(f"Image: {img_name} -> Age: {age}, Group: {age_group}, Gender: {gender}")

                # Save face in the corresponding folder
                dest_folder = os.path.join(output_path, age_group, gender)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, f"{age}_{gender}_{img_name}")
                copy2(img_path, dest_path)

                log_text.insert(tk.END, f"{img_name} -> Age: {age}, Group: {age_group}, Gender: {gender}\n")
                root.update()

    messagebox.showinfo("Completed", f"Processing completed! Total images processed: {total_images}")

# UI Layout
tk.Label(root, text="Input Folder:").pack()
tk.Entry(root, textvariable=input_folder, width=50).pack()
tk.Button(root, text="Browse", command=select_input_folder).pack()

tk.Label(root, text="Output Folder:").pack()
tk.Entry(root, textvariable=output_folder, width=50).pack()
tk.Button(root, text="Browse", command=select_output_folder).pack()

# GPU Memory Limit Entry
tk.Label(root, text="GPU Memory Limit (MB):").pack()
tk.Entry(root, textvariable=gpu_memory_limit, width=10).pack()  # 🔥 Entry instead of slider

# Batch Size Control
tk.Label(root, text="Batch Size:").pack()
batch_slider = tk.Scale(root, from_=1, to=4, orient="horizontal", variable=batch_size)  
batch_slider.pack()

# GPU/CPU Toggle
tk.Label(root, text="Processing Mode:").pack()
gpu_checkbox = tk.Checkbutton(root, text="Use GPU", variable=gpu_enabled)
gpu_checkbox.pack()

# Start & Stop Buttons
tk.Button(root, text="Start Processing", command=process_images, bg="green", fg="white").pack(pady=5)
tk.Button(root, text="Stop Processing", command=stop_process, bg="red", fg="white").pack(pady=5)

log_text = scrolledtext.ScrolledText(root, width=70, height=10)
log_text.pack()

root.mainloop()
