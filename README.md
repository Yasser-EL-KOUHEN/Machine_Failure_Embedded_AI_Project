

```markdown
# PRACTICAL SESSION 1 — Deep Learning for Predictive Maintenance

This report details a deep learning project for predictive maintenance using industrial sensor data. The objective is to develop a model that can predict whether a machine will fail and, if so, determine the specific type of failure. The project uses the AI4I 2020 Predictive Maintenance Dataset, which comprises 10,000 records of sensor data.

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Background and Motivation](#11-background-and-motivation)
  - [1.2 Problem Statement](#12-problem-statement)
- [2. Dataset Overview](#2-dataset-overview)
  - [2.1 Description of Features](#21-description-of-features)
  - [2.2 Labels and Failure Types](#22-labels-and-failure-types)
  - [2.3 Data Source and Access](#23-data-source-and-access)
- [3. Project Objectives](#3-project-objectives)
- [4. Methodology and Design Decisions](#4-methodology-and-design-decisions)
  - [4.1 Exploratory Data Analysis (EDA)](#41-exploratory-data-analysis-eda)
    - [4.1.1 Visualizing the Data Distribution](#411-visualizing-the-data-distribution)
    - [4.1.2 Observations on Class Imbalance](#412-observations-on-class-imbalance)
  - [4.2 Data Preprocessing](#42-data-preprocessing)
    - [4.2.1 Feature Selection and Encoding](#421-feature-selection-and-encoding)
    - [4.2.2 Data Normalization and Splitting](#422-data-normalization-and-splitting)
  - [4.3 Initial Modeling Without Balancing](#43-initial-modeling-without-balancing)
    - [4.3.1 Model Architecture](#431-model-architecture)
    - [4.3.2 Training Setup and Hyperparameters](#432-training-setup-and-hyperparameters)
    - [4.3.3 Baseline Performance Analysis](#433-baseline-performance-analysis)
  - [4.4 Modeling With Balancing Techniques](#44-modeling-with-balancing-techniques)
    - [4.4.1 Identification of Class Imbalance](#441-identification-of-class-imbalance)
    - [4.4.2 Resampling Strategy: SMOTE](#442-resampling-strategy-smote)
    - [4.4.3 Aggregation of Failure Types](#443-aggregation-of-failure-types)
    - [4.4.4 Separate Binary and Multi-Class Models](#444-separate-binary-and-multi-class-models)
  - [4.5 Model Architecture Justification](#45-model-architecture-justification)
- [5. Training, Evaluation, and Results](#5-training-evaluation-and-results)
  - [5.1 Training Curves and Convergence](#51-training-curves-and-convergence)
  - [5.2 Evaluation Metrics](#52-evaluation-metrics)
    - [5.2.1 Confusion Matrix and Classification Report](#521-confusion-matrix-and-classification-report)
    - [5.2.2 ROC Curves and AUC Analysis](#522-roc-curves-and-auc-analysis)
    - [5.2.3 Feature Importance Analysis](#523-feature-importance-analysis)
  - [5.3 Observations and Discussion](#53-observations-and-discussion)
- [6. Conclusions and Future Work](#6-conclusions-and-future-work)
- [7. Instructions for Execution](#7-instructions-for-execution)
- [8. References](#8-references)
- [9. Authors and License](#9-authors-and-license)

---

## 1. Introduction

### 1.1 Background and Motivation

In industrial settings, unexpected machine failures can lead to significant downtime and financial loss. Predictive maintenance aims to preemptively detect faults, enabling proactive repairs. Leveraging deep learning to interpret sensor data can substantially enhance maintenance efficiency and reduce operational risks.

### 1.2 Problem Statement

The primary challenge is to develop a robust predictive model that:
1. **Predicts if a machine will fail** (binary classification).
2. **Identifies the specific failure type** when a failure occurs (multi-class classification).

The complexity of the problem is heightened by the highly imbalanced dataset, where non-failure cases dominate and certain failure types are extremely rare.

---

## 2. Dataset Overview

### 2.1 Description of Features

The dataset contains industrial sensor data with the following key features:
- **Air temperature [K]**: Ambient temperature measurement.
- **Process temperature [K]**: Temperature within the machine process.
- **Rotational speed [rpm]**: Speed of the machine’s rotating components.
- **Torque [Nm]**: The applied rotational force.
- **Tool wear [min]**: Duration or extent of tool degradation.
- **Type**: Categorical variable indicating the machine or product type (requires encoding).

### 2.2 Labels and Failure Types

Two types of labels are provided:
- **Machine failure**: A binary indicator (0 for no failure, 1 for failure).
- **Failure Types**: Five possible types of failure:
  - **TWF**: Tool Wear Failure
  - **HDF**: Heat Dissipation Failure
  - **PWF**: Power Failure
  - **OSF**: Overstrain Failure
  - **RNF**: Random Failure

### 2.3 Data Source and Access

The dataset is stored as a CSV file named `ai4i2020.csv` available on the eCAMPUS platform. In our experiments, we load the dataset using Google Drive in a Colab environment.

---

## 3. Project Objectives

- **Exploratory Data Analysis (EDA):** Understand the structure and distribution of the data, including visualization of imbalanced classes.
- **Data Preprocessing:** Select relevant features, encode categorical variables, and normalize sensor readings.
- **Model Development:** Create deep learning models for both binary and multi-class classification.
- **Balancing Techniques:** Address severe class imbalance using oversampling methods (SMOTE) and analyze their impact.
- **Evaluation:** Use robust metrics (confusion matrices, ROC curves, classification reports) to assess model performance.
- **Interpretation:** Justify design choices and provide insights into feature importance and limitations.

---

## 4. Methodology and Design Decisions

### 4.1 Exploratory Data Analysis (EDA)

#### 4.1.1 Visualizing the Data Distribution

- **Initial Inspection:**  
  The dataset is loaded into a Pandas DataFrame, and the first few rows along with the overall shape are printed.
  
- **Bar Plots and Histograms:**  
  - A bar plot of `Machine failure` shows the overwhelming number of non-failure cases.
  - Separate bar charts for each failure type (TWF, HDF, PWF, OSF, RNF) are generated.
  - For machines that have failed, an additional category "No Specific Failure" is added to identify cases where a failure occurred but no type was flagged.

#### 4.1.2 Observations on Class Imbalance

- **Findings:**  
  - Non-failure instances constitute the majority of the data.
  - Among the failure types, some classes (e.g., TWF, RNF) are extremely underrepresented.
  
- **Implications:**  
  This imbalance will likely cause the model to be biased toward the majority class, reducing its sensitivity (recall) for rare failures.

### 4.2 Data Preprocessing

#### 4.2.1 Feature Selection and Encoding

- **Selected Features:**  
  We use key sensor measurements such as "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", and the encoded "Type" variable.
  
- **Encoding:**  
  The categorical feature `Type` is transformed into numerical values using label encoding.

#### 4.2.2 Data Normalization and Splitting

- **Normalization:**  
  StandardScaler is applied to ensure that all features contribute equally during training.
  
- **Train-Test Split:**  
  The data is split into training (80%) and testing (20%) sets to ensure unbiased evaluation.

### 4.3 Initial Modeling Without Balancing

#### 4.3.1 Model Architecture

- **Dual-Output Model:**  
  A deep neural network is constructed using the Keras Functional API with two outputs:
  - **Output 1:** A sigmoid activation for binary classification of `Machine failure`.
  - **Output 2:** A sigmoid (or softmax) activation for predicting multiple failure type flags.

#### 4.3.2 Training Setup and Hyperparameters

- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss Function:** Binary crossentropy applied to both outputs.
- **Epochs & Batch Size:** The model is trained for 50 epochs with a batch size of 32.
  
#### 4.3.3 Baseline Performance Analysis

- **Results:**  
  Despite a high overall accuracy, the model exhibits low recall for failure cases due to the overwhelming majority of non-failure examples.
  
- **Conclusion:**  
  The baseline model highlights the need for balancing the dataset to improve minority class prediction.

### 4.4 Modeling With Balancing Techniques

#### 4.4.1 Identification of Class Imbalance

- **Issue:**  
  The dataset has a marked imbalance, with only a small fraction of examples corresponding to machine failures and, within failures, some types are extremely rare.
  
#### 4.4.2 Resampling Strategy: SMOTE

- **SMOTE Implementation:**  
  Synthetic Minority Over-sampling Technique (SMOTE) is used to generate synthetic samples for the underrepresented failure types. This is applied only on the training set to avoid leakage.
  
- **Rationale:**  
  SMOTE helps in creating a more balanced distribution, allowing the model to learn better representations for minority classes.

#### 4.4.3 Aggregation of Failure Types

- **New Label – `Failure_Type`:**  
  Instead of using multiple binary columns, the failure type information is aggregated into a single categorical column. Priority is given to the first failure type found in the record.
  
- **One-Hot Encoding:**  
  The aggregated `Failure_Type` is then transformed using one-hot encoding for multi-class classification.

#### 4.4.4 Separate Binary and Multi-Class Models

- **Binary Model:**  
  A separate model is developed to predict "No Machine failure" (binary output) using a sigmoid activation.
  
- **Multi-Class Model:**  
  Another model is built to classify the specific failure type using a softmax activation in the final layer.
  
- **Justification:**  
  This separation allows each model to specialize, improving overall performance and providing clearer insights into each task.

### 4.5 Model Architecture Justification

- **Layer Design:**  
  Both models use a similar architecture with several dense layers, Batch Normalization, and Dropout:
  - **Dense Layers:** Extract and transform feature representations.
  - **Batch Normalization:** Stabilizes and accelerates training.
  - **Dropout:** Mitigates overfitting by randomly deactivating neurons during training.
  
- **Activation Functions:**  
  - **ReLU:** Used in hidden layers for non-linear transformations.
  - **Sigmoid/Softmax:** Applied at the output layers for probability estimation in binary and multi-class tasks respectively.
  
- **Overall Rationale:**  
  This architecture is designed to be deep enough to capture complex patterns while remaining regularized to avoid overfitting. The choice of SMOTE and the dual-model strategy address the imbalanced nature of the dataset.

---

## 5. Training, Evaluation, and Results

### 5.1 Training Curves and Convergence

- **Monitoring:**  
  Training and validation accuracy/loss curves are plotted over 50 epochs.
  
- **Observations:**  
  Both models demonstrate smooth convergence with no significant gap between training and validation performance, indicating that overfitting is minimal.

### 5.2 Evaluation Metrics

#### 5.2.1 Confusion Matrix and Classification Report

- **Binary Classifier:**  
  The confusion matrix shows high true negative rates and an acceptable true positive rate, while the classification report provides precision, recall, and F1-scores.
  
- **Multi-Class Classifier:**  
  For failure types, confusion matrices are generated per class. The classification report details performance per failure type, highlighting challenges for extremely rare classes (e.g., TWF).

#### 5.2.2 ROC Curves and AUC Analysis

- **Binary ROC Curve:**  
  The ROC curve for the binary model displays an AUC near 0.98, indicating excellent discrimination.
  
- **Multi-Class ROC Curves:**  
  ROC curves for each failure type (one-vs-rest) are plotted. Most classes show high AUC values, though the rarest class may have a lower AUC, reflecting prediction challenges.

#### 5.2.3 Feature Importance Analysis

- **Method:**  
  The weights of the first dense layer are used to compute an average importance score for each feature.
  
- **Findings:**  
  Features such as rotational speed, air temperature, and torque are among the most influential, while the encoded machine type shows lower importance.
  
### 5.3 Observations and Discussion

- **Without Balancing:**  
  The model’s high overall accuracy masks poor sensitivity to failure cases due to the imbalance.
  
- **With Balancing (SMOTE):**  
  The balanced models achieve:
  - **Binary Model:** Improved detection of failure cases with a slight trade-off in precision.
  - **Multi-Class Model:** Good performance on common failure types (HDF, OSF, PWF) with persistent challenges in predicting the rarest type (TWF).
  
- **Conclusion:**  
  Balancing the dataset is crucial. Although SMOTE significantly improves recall for minority classes, further improvements (e.g., alternative resampling strategies, feature engineering) may be necessary for the most challenging classes.

---

## 6. Conclusions and Future Work

- **Conclusions:**
  - The initial model without balancing confirmed that high overall accuracy can be misleading when classes are imbalanced.
  - SMOTE-based balancing and the development of separate binary and multi-class models substantially improved performance, especially for failure detection.
  - Both models showed stable training and good generalization as evidenced by convergence curves and evaluation metrics.
  
- **Limitations:**
  - Extremely rare failure types (e.g., TWF) remain difficult to predict accurately.
  - Synthetic samples may not fully capture the diversity of real-world failure conditions.
  
- **Future Work:**
  - Explore additional resampling methods (e.g., ADASYN, SMOTE-ENN) to further enhance minority class representation.
  - Investigate advanced feature engineering and selection techniques.
  - Consider alternative model architectures (such as recurrent neural networks) if temporal data becomes available.
  - Experiment with ensemble methods to potentially boost overall performance.

---

## 7. Instructions for Execution

1. **Environment Setup:**  
   Ensure that Python 3.x is installed and set up an environment (e.g., Google Colab). Install necessary packages using:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn imbalanced-learn
   ```

2. **Data Access:**  
   Place the `ai4i2020.csv` file in an accessible location (e.g., mount Google Drive in Colab) and update the dataset path accordingly in the code.

3. **Execution Flow:**  
   - **Step 1:** Load the dataset and perform Exploratory Data Analysis (EDA) to understand data distribution.
   - **Step 2:** Preprocess the data by selecting features, encoding categorical variables, and normalizing values.
   - **Step 3:** Split the dataset into training and testing sets.
   - **Step 4:** Train the initial dual-output deep learning model without balancing.
   - **Step 5:** Apply SMOTE for balancing the training set and develop separate models for binary and multi-class classification.
   - **Step 6:** Evaluate the models using confusion matrices, classification reports, ROC curves, and feature importance analysis.
   - **Step 7:** Review training curves to ensure model stability and check for overfitting.

4. **Reproducibility:**  
   For reproducible results, set random seeds for TensorFlow, NumPy, and other libraries as needed.

---

## 8. References

- [Scikit-learn: Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Scikit-learn: ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
- [Imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [TensorFlow Tutorial on Imbalanced Data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=fr)

---

## 9. Authors and License

**Author:** Yasser EL KOUHEN  
**Supervision:** Professor Kévin HECTOR, Mines Saint-Étienne  
```

---

This version of the README.md includes all the comprehensive details along with proper attribution in Parts 8 and 9. Adjust any section further as needed for your project.
