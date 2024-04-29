# services/data_preparation.py

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def prepare_training_data(csv_file, img_size=(224, 224)):
    # Load training data from CSV
    train_data = pd.read_csv('CNN_Model_Train_Data.csv')

    # Define image size and labels
    X = []
    y = []

    # Load and preprocess images
    for index, row in train_data.iterrows():
        image_path = row['image_path']  # Assuming 'image_path' column contains file paths
        label = row['label']  # Assuming 'label' column contains product labels

        # Load image and resize
        image = load_img(image_path, target_size=img_size)
        image = img_to_array(image)

        # Normalize pixel values
        image = image / 255.0

        X.append(image)
        y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
