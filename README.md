# 🧠 Handwritten Digit Recognition using Neural Network & ML

This project is a machine learning solution that recognizes handwritten digits (0–9) using the **Dataset** and a **Neural Network model** built with TensorFlow/Keras. It covers all the fundamental steps of building, training, evaluating, and making predictions with an ML model.

---

## 📌 Project Overview

- **Goal**: Accurately classify handwritten digits.
- **Dataset**: Train.csv
- **Tech Stack**: Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn
- **Model**: Fully Connected Neural Network (Multi-layer Perceptron)

---

## 🗂️ Project Structure

```
Handwritten-Digit-Recognition/
├── src/
│   └── main.py                 
├── notebooks/                  
│   └── handwritten-digits.ipynb
├── README.md
├── requirements.txt
```

---

## 🚀 Steps Implemented in Code

### ✅ Step 1: Import Libraries
Essential libraries like `tensorflow`, `numpy`, `matplotlib`, and `sklearn` are imported.

### ✅ Step 2: Load and Explore the Dataset
Load the dataset using Pandas. Display sample images and check input shapes.

### ✅ Step 3: Preprocess the Data
- Normalize image pixel values (0–255 → 0–1)
- Reshape data as needed

### ✅ Step 4: One-Hot Encode the Labels
Convert class labels (0–9) to one-hot encoded vectors (optional for softmax output).

### ✅ Step 5: Split the Data
Split the training dataset into training and validation sets (e.g., 80/20).

### ✅ Step 6: Build the Neural Network Model
Build a simple feedforward neural network

### ✅ Step 7: Train the Model
Train the model using the `fit()` function for a fixed number of epochs.

### ✅ Step 8: Evaluate the Model
Evaluate the model on the test set and print accuracy.

### ✅ Step 9: Make Predictions
Make predictions using the `predict()` function and visualize results.

---

## 🛠️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/saiganesh0084/Handwritten-Digit-Recognition-using-Neural-Network-and-Machine-Learning.git
cd Handwritten-Digit-Recognition-using-Neural-Network-and-Machine_learning
```

### 2. Create Virtual Environment (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
```bash
python src/main.py
```

---

## 📊 Sample Output

```
Epoch 1/5
loss: 0.30 - accuracy: 0.91
...
Test Accuracy: 97.82%
```

You can also visualize prediction results with Matplotlib to display digits and predicted labels.

---

## 📁 Requirements

See `requirements.txt`:
```
tensorflow
numpy
matplotlib
scikit-learn
```

---

## 📌 Future Improvements

- Add convolutional neural network (CNN) for better accuracy
- Integrate model into a simple GUI or web app using Streamlit
- Add confusion matrix visualization

---
## Contact
- saiganeshganoju@gmail.com
