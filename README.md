# DeepHAR: Human Activity Recognition

### **The Motivation**
In traditional Human Activity Recognition (HAR), achieving high accuracy typically requires hundreds of heavily engineered mathematical features (Fourier transforms, signal filters, etc.) carefully crafted by human experts. 

The main motivation of this project is to compare manual feature engineering  and and a freshly build a 1D Convolutional Neural Network (CNN) in PyTorch that can learn the pure physics of human movement from scratch. 

### **The Dataset: UCI HAR**
This project utilizes the well-known **UCI Human Activity Recognition Using Smartphones** dataset. 
* **The Subjects:** 30 volunteers performing 6 baseline activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying).
* **The Hardware:** A smartphone worn on the waist capturing 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz.
* **The Format:** The raw inertial signals are sliced into 2.56-second overlapping windows (128 time steps per window) across 9 distinct physical channels.

### **The Tech Stack & Toolkit**
This project bridges the gap between traditional data science diagnostics and modern deep learning architecture.
* **Deep Learning:** `torch`, `torch.nn`, `torch.optim` (PyTorch for custom DataLoaders, 1D CNN architecture, and backpropagation).
* **Data Processing:** `numpy`, `pandas` (For complex 3D tensor reshaping and channel stacking).
* **Machine Learning & Diagnostics:** `scikit-learn` (For t-SNE dimensionality reduction, classification metrics, and confusion matrices).
* **Visualization:** `matplotlib`, `seaborn` (For plotting learning curves and tracking the Sitting/Standing overlap).

### **Repository Structure**
To run this project, ensure your local directory matches the following structure.

```text
├── UCIHARdata/                  # The core dataset directory
│   ├── test/                    # Unseen evaluation data
│   │   ├── Inertial Signals/    # Raw 128-step time-series windows (x, y, z)
│   │   ├── subject_test.txt     # Volunteer IDs
│   │   ├── X_test.txt           # Pre-engineered static features (for baseline)
│   │   └── y_test.txt           # True activity labels
│   ├── train/                   # Training data
│   │   ├── Inertial Signals/    
│   │   ├── subject_train.txt    
│   │   ├── X_train.txt          
│   │   └── y_train.txt          
│   ├── activity_labels.txt      # Master list of the 6 activities
│   ├── features_info.txt        # Documentation of variables
│   ├── features.txt             # Names of the 561 static features
│   └── README.txt               # Original dataset documentation
│
└── DeepHAR_main.ipynb           # The main Jupyter Notebook containing all code
