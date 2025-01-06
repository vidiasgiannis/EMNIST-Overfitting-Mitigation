# **EMNIST Overfitting Mitigation**

This project investigates the **problem of overfitting** in neural networks by experimenting with regularization techniques such as **Dropout**, **L1/L2 penalties**, and **Label Smoothing**. We train and evaluate different models on the **EMNIST Balanced dataset** to analyze their generalization performance and overfitting mitigation.

---


## **1. Dataset Information**

### **Download the Dataset**
The dataset used is **EMNIST Balanced** (47 classes, 28x28 grayscale images):
1. Download the dataset from [NIST EMNIST Dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset).
2. Extract the dataset to the `data/` directory:
   ```bash
   unzip emnist-balanced-dataset.zip -d ./data/
   ```
   Final structure:
   ```
   /data/
     └── emnist-balanced/
         ├── training_images.npy
         ├── training_labels.npy
         ├── test_images.npy
         └── test_labels.npy
   ```

---

## **3. Installation**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/vidiasgiannis/EMNIST-Overfitting-Mitigation.git
cd EMNIST-Overfitting-Mitigation
```

### **Step 2: Create a Virtual Environment**
```bash
python -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **4. Run Experiments**

### **1. Train the Baseline Model**
```bash
python src/train_baseline.py
```

### **2. Train with Dropout Regularization**
```bash
python src/train_with_dropout.py --dropout 0.5
```
Adjust the `--dropout` value to control the inclusion probability (e.g., 0.85).

### **3. Train with L1/L2 Penalty**
```bash
python src/train_with_penalty.py --l1_penalty 0.001
python src/train_with_penalty.py --l2_penalty 0.001
```

---

## **5. Results Overview**

### **Baseline Performance**:
The model was trained with 3 hidden layers (128 neurons each) and evaluated for overfitting using the EMNIST dataset:
- Training accuracy: **~95%**.
- Validation accuracy: **~81%**.
- Observed **generalization gap**: The difference between training and validation errors.

### **Regularization Results**:
- **Dropout**: Improved validation accuracy to **85.1%** with a **0.85** inclusion rate.
- **L2 Penalty**: Achieved a validation accuracy of **84.9%** with a penalty of **0.001**.
- **Label Smoothing**: Prevented overconfidence, improving the generalization of the model.

---
