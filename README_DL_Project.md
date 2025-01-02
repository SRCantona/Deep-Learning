
# Deep Learning Assignment - Parallel Processing and Machine Learning

This project demonstrates the application of parallel computing in a deep learning environment, focusing on race conditions, synchronization, and performance optimization.

The project explores the prediction of data through various models and optimization techniques.

---

## Key Concepts Addressed:

### 1. Race Condition in Parallel Processing
A race condition occurs when multiple threads attempt to modify shared resources concurrently without proper synchronization.

**Example of Race Condition in Python:**

```python
gold_price = 1500

def increment_price():
    global gold_price
    for _ in range(100000):
        gold_price += 1

def decrement_price():
    global gold_price
    for _ in range(100000):
        gold_price -= 1
```

In the above example, the `gold_price` variable is accessed by two functions simultaneously, leading to inconsistent results.

---

### 2. Synchronization with Locks

To resolve race conditions, the `threading.Lock` mechanism is employed:

```python
from threading import Lock

lock = Lock()

def increment_price_safe():
    global gold_price
    with lock:
        for _ in range(100000):
            gold_price += 1
```

This ensures that only one thread can modify the variable at a time, preserving data integrity.

---

## Notebook Breakdown

### Section 1
# *Iris* Data Classification (Using TensorFlow)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEn
```
---

### Section 2
# **imports**

```python
a = pd.read_csv('/content/iris.csv')
iris = pd.DataFrame(a)
iris.head()
```
---

### Section 3
# **dataset**

```python
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Encode target labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_tran
```
---

### Section 4

# **Data Preprocessing**
# This section prepares the dataset for training by:
# 1. Separating features (X) and labels (y).
# 2. Encoding the categorical target labels into numerical format.
# 3. Applying one-hot encoding for multi-class classification compatibility.
# 4. Scaling features for unifor

```python

def create_model(activation):
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation=activation),  # First hidden layer
  
```
---

### Section 5
## Model Development

# 1. Define a Sequential Model.
# 2. Add Layers(Hidden Layer, Output Layer).
# 3. Compile the Model.


```python


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
    history = model.fit(
        X_train, y_train,
        validat
```
---

### Section 6
# **Model Training**
# in this section want to fit the model
# 1. Training Data: (X_train, y_train), used for model optimization.
# 2. Validation Data: (X_val, y_val), used to monitor performance on unseen data during training.
# 3. Number of Epochs: increase epochs = increase Validation Accuracy.
#

```python

# evaluate the model
def evaluate_model(model, X_val, y_val):
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    return val_los
```
---

### Section 7
# **Model Evaluation**

# 1. evaluate a trained model using the validation dataset (X_val, y_val) and calculate its loss and accuracy.
# 2. display the performance metrics and training history for a specific activation function.
# 3. Prints the validation loss and accuracy for the model trained with

```python
# Function to display output results
def display_results(activation, history, val_loss, val_accuracy):
    print(f"\nResults for model with {activatio
```
---

### Section 8

# **Reflection**

# This project was an insightful journey into building a neural network for classifying the Iris dataset. It involved essential steps like preparing the data through scaling and encoding, designing a sequential model, and experimenting with activation functions such as relu, sigmo

```python
# List of activation functions
activations = ['relu', 'sigmoid', 'tanh', 'linear']

# Loop through each activation function
for activation in activati
```
---
