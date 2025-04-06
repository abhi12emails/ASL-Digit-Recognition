#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# ## Step 1: Extract Keypoints Using MediaPipe Hands

# ## Step 2: Load and Process the Dataset

# In[2]:


import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    keypoints = np.zeros((21, 3))  # 21 keypoints, each with (x, y, z)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[i] = [lm.x, lm.y, lm.z]
    
    return keypoints

# Define Path
dataset_path = r"C:\Users\rohin\Downloads\Sign-Language-Digits-Dataset-master\Sign-Language-Digits-Dataset-master\Dataset"

X = []  # Initialize as empty list
y = []  # Initialize as empty list
num_classes = 10  # Digits 0-9

# Load Dataset
for digit in range(num_classes):
    digit_path = os.path.join(dataset_path, str(digit))
    
    if not os.path.exists(digit_path):  # Check if folder exists
        print(f"Warning: Folder {digit_path} not found!")
        continue  # Skip if folder does not exist
    
    for img_name in os.listdir(digit_path):
        img_path = os.path.join(digit_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue  # Skip unreadable images

        keypoints = extract_keypoints(image)
        X.append(keypoints)  # Shape (21, 3)
        y.append(digit)

# Convert lists to numpy arrays
X = np.array(X)  # Shape should be (num_samples, 21, 3)
y = to_categorical(np.array(y), num_classes=num_classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset processed successfully! X shape: {X.shape}, y shape: {y.shape}")


# In[3]:


import matplotlib.pyplot as plt
import cv2

# Create a figure with subplots (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Store already selected digits
sampled_digits = set()

for digit in range(num_classes):
    digit_path = os.path.join(dataset_path, str(digit))
    
    if not os.path.exists(digit_path):
        print(f"Warning: Folder {digit_path} not found!")
        continue

    # Get a list of image files in the digit folder
    img_files = os.listdir(digit_path)
    
    if not img_files:
        print(f"Warning: No images found for digit {digit}")
        continue

    # Select the first image in the folder
    img_path = os.path.join(digit_path, img_files[0])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    
    if image is None:
        print(f"Warning: Could not read {img_path}")
        continue
    
    # Plot the image in its corresponding position
    ax = axes[digit // 5, digit % 5]
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Digit: {digit}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# In[4]:


import matplotlib.pyplot as plt

# Define a function to plot keypoints
def plot_keypoints(keypoints, label):
    keypoints = np.array(keypoints)  # Ensure it's a NumPy array
    plt.scatter(keypoints[:, 0], keypoints[:, 1], marker='o', c='red', label=f'Digit: {label}')
    
    # Connect keypoints to show hand structure
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    for connection in connections:
        pt1, pt2 = connection
        plt.plot([keypoints[pt1, 0], keypoints[pt2, 0]], 
                 [keypoints[pt1, 1], keypoints[pt2, 1]], 'r')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Digit: {label}")
    plt.gca().invert_yaxis()  # Invert Y-axis for correct orientation

# Get one sample per digit
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns
sampled_digits = set()

for i, (keypoints, label) in enumerate(zip(X, np.argmax(y, axis=1))):
    if label not in sampled_digits:
        sampled_digits.add(label)
        ax = axes[label // 5, label % 5]  # Arrange in grid
        plt.sca(ax)  # Set current axis
        plot_keypoints(keypoints, label)
    
    if len(sampled_digits) == num_classes:  # Stop after finding one per digit
        break

plt.tight_layout()
plt.show()


# ## Step 3: Build the CNN + Attention LSTM Model

# In[9]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, LSTM, Dropout, TimeDistributed, Attention, GlobalAveragePooling1D

def create_cnn_attention_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # CNN Layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Reshape for LSTM
    x = TimeDistributed(Dense(128, activation='relu'))(x)  
    
    # LSTM Layer (Make sure it outputs a single vector)
    lstm_out = LSTM(128, return_sequences=True)(x)
    
    # Apply Attention Mechanism
    attention_out = Attention()([lstm_out, lstm_out])  
    attention_out = GlobalAveragePooling1D()(attention_out)  # Reduce dimensionality
    
    # Fully Connected Layers
    dense_out = Dense(128, activation='relu')(attention_out)
    dense_out = Dropout(0.5)(dense_out)
    outputs = Dense(num_classes, activation='softmax')(dense_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Define input shape and number of classes
input_shape = (21, 3)  # Example: 21 keypoints, 3 coordinates (x, y, z)
num_classes = 10  # Digits 0-9

# Create and summarize model
model = create_cnn_attention_lstm_model(input_shape, num_classes)
model.summary()


# ## Step 4: Train the Model

# In[10]:


# Define training parameters
epochs = 20  # Adjust as needed

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=32
)


# In[11]:


# Save the Model
model.save("sign_language_cnn_attentionlstm.h5")
print("Model saved successfully!")

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predict on test set
y_pred = model.predict(X_test)

# Convert predictions & actual labels to class indices
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print Classification Report
print("\n Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Print Confusion Matrix
print("\n Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))


# ## Step 5: Real-time Prediction from Webcam

# In[13]:


# Real-Time Prediction with Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    keypoints = extract_keypoints(frame)
    keypoints = np.expand_dims(keypoints, axis=0)  # Reshape for model input
    
    prediction = model.predict(keypoints)
    digit = np.argmax(prediction)

    # Display Prediction on Webcam
    cv2.putText(frame, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)  # Black & Bold
    cv2.putText(frame, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White Outline

    cv2.imshow('Sign Language Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ## Real-Time Demo

# In[14]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained CNN + BiLSTM model
model = load_model("sign_language_cnn_attentionlstm.h5")
print("Model Loaded Successfully!")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to Extract Keypoints
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    keypoints = np.zeros((21, 3))  # 21 keypoints, each with (x, y, z)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[i] = [lm.x, lm.y, lm.z]
    return keypoints  # Return 21x3 array (not flattened)

# Start Webcam for Real-Time Prediction
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    keypoints = extract_keypoints(frame)
    keypoints = np.expand_dims(keypoints, axis=0)  # Reshape for model input

    # Predict the Digit
    prediction = model.predict(keypoints)
    digit = np.argmax(prediction)

    # Display Prediction on Webcam (Bold Black with White Outline)
    cv2.putText(frame, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)  # Black & Bold
    cv2.putText(frame, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White Outline

    cv2.imshow('Sign Language Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




