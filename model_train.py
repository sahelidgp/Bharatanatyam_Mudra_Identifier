# model_train.py
"""
Usage:
    python model_train.py --dataset_dir /path/to/Bharatnatyam_mudra_dataset --output_dir ./model_artifacts

This script:
1. Walks the dataset directory where each subfolder is a class (mudra).
2. Extracts MediaPipe hand landmarks (21 landmarks * 3 coords = 63 features).
3. Trains a small Keras Dense classifier on the extracted landmarks.
4. Saves the trained model (model.h5), data scaler (scaler.pkl), and 
   label encoder (label_encoder.pkl) into the specified output directory.
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

def extract_landmarks_from_image(image_bgr):
    """
    Extract 21 hand landmarks (x, y, z) flattened into a shape of (63,).
    Returns a NumPy array of shape (63,) or None if no hand is detected.
    """
    with mp_hands.Hands(static_image_mode=True,
                          max_num_hands=1,
                          min_detection_confidence=0.5) as hands:
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        res = hands.process(image_rgb)
        
        if res.multi_hand_landmarks:
            # Get landmarks for the first detected hand
            lm = res.multi_hand_landmarks[0]
            coords = []
            for point in lm.landmark:
                coords.extend([point.x, point.y, point.z])
            return np.array(coords, dtype=np.float32)
    return None

def build_dataset(dataset_dir, verbose=True):
    """
    Walks through the dataset directory, extracts landmarks from each image,
    and returns the feature vectors (X) and corresponding labels (y).
    """
    X = []
    y = []
    
    # Get sorted list of class names (subdirectories)
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if verbose:
        print("Found classes:", classes)
        
    for cls in classes:
        folder = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for f in tqdm(files, desc=f"Processing {cls}", leave=False):
            path = os.path.join(folder, f)
            img = cv2.imread(path)
            if img is None:
                if verbose:
                    print(f"Warning: Could not read image {path}")
                continue
            
            lm = extract_landmarks_from_image(img)
            if lm is not None:
                X.append(lm)
                y.append(cls)
                
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"Total examples with detected hand: {len(X)}")
        
    return X, y

def build_model(input_dim, num_classes):
    """
    Builds and compiles a simple Keras sequential model for classification.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Building dataset from images...")
    X, y = build_dataset(dataset_dir)
    if len(X) == 0:
        raise RuntimeError("No hand landmarks were extracted. Please check your dataset images and MediaPipe detection.")

    # Encode string labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Shuffle the dataset thoroughly
    idx = np.arange(len(X))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(idx)
    X = X[idx]
    y_enc = y_enc[idx]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    # Scale the feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build and summarize the model
    num_classes = len(le.classes_)
    input_dim = X_train_scaled.shape[1]
    model = build_model(input_dim, num_classes)
    model.summary()

    # Define callbacks for training
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    print("\nStarting model training...")
    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_val_scaled, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=cb,
                        verbose=1)

    # Save the essential artifacts for the web app
    model_path = os.path.join(output_dir, 'model.h5')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    le_path = os.path.join(output_dir, 'label_encoder.pkl')

    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    loss, acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"\nTraining complete.")
    print(f"Saved artifacts to {output_dir}.")
    print(f"Final Validation Loss: {loss:.4f}, Final Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a mudra classification model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root directory of the image dataset (with subfolders per class).")
    parser.add_argument("--output_dir", type=str, default="./model_artifacts", help="Directory to save the trained model and other artifacts.")
    args = parser.parse_args()
    main(args)