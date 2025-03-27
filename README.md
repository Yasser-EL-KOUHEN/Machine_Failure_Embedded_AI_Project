
# PRACTICAL SESSION 1 — Deep Learning for Predictive Maintenance & Embedded AI Deployment

This project details an end‑to‑end deep learning solution for predictive maintenance using industrial sensor data, and its subsequent deployment on an STM32L4R9 microcontroller. The objective is to develop a robust model that can predict whether a machine will fail and, if so, determine the specific type of failure. In addition to the Colab‑based training and evaluation (including the use of SMOTE to balance a highly imbalanced dataset), the project covers:
- Exporting the trained Keras model to TensorFlow Lite (TFLite) format.
- Integrating the TFLite model with an STM32 board via STM32CubeIDE.
- Communicating with the board using a Python script over UART.
- Documenting challenges related to the board’s compatibility with the application.

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
- [6. Embedded AI Deployment](#6-embedded-ai-deployment)
  - [6.1 Model Conversion and Export](#61-model-conversion-and-export)
  - [6.2 Integration on STM32L4R9](#62-integration-on-stm32l4r9)
    - [6.2.1 C Code Modifications](#621-c-code-modifications)
    - [6.2.2 Communication via UART (Python Script)](#622-communication-via-uart-python-script)
  - [6.3 Encountered Challenges](#63-encountered-challenges)
- [7. Conclusions and Future Work](#7-conclusions-and-future-work)
- [8. Instructions for Execution](#8-instructions-for-execution)
- [9. References](#9-references)
- [10. Authors and License](#10-authors-and-license)

---

## 1. Introduction

### 1.1 Background and Motivation

In industrial settings, unexpected machine failures can lead to significant downtime and financial loss. Predictive maintenance aims to preemptively detect faults, enabling proactive repairs. Leveraging deep learning to interpret sensor data can substantially enhance maintenance efficiency and reduce operational risks.

### 1.2 Problem Statement

The primary challenge is to develop a robust predictive model that:
1. **Predicts if a machine will fail** (binary classification).
2. **Identifies the specific failure type** when a failure occurs (multi-class classification).

The complexity of the problem is heightened by the highly imbalanced dataset, where non‑failure cases dominate and certain failure types are extremely rare.

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

The dataset is stored as a CSV file named `ai4i2020.csv` available on the eCAMPUS platform. In our experiments, we loaded the dataset in Google Colab by mounting Google Drive.

---

## 3. Project Objectives

- **Exploratory Data Analysis (EDA):** Understand the structure and distribution of the data, including visualization of imbalanced classes.
- **Data Preprocessing:** Select relevant features, encode categorical variables, and normalize sensor readings.
- **Model Development:** Create deep learning models for both binary and multi‑class classification.
- **Balancing Techniques:** Address severe class imbalance using oversampling methods (SMOTE) and analyze their impact.
- **Evaluation:** Use robust metrics (confusion matrices, ROC curves, classification reports) to assess model performance.
- **Embedded Deployment:** Convert and deploy the trained model to an STM32L4R9 board and establish communication over UART.
- **Interpretation:** Justify design choices and provide insights into feature importance and limitations.

---

## 4. Methodology and Design Decisions

### 4.1 Exploratory Data Analysis (EDA)

#### 4.1.1 Visualizing the Data Distribution

- **Initial Inspection:**  
  The dataset is loaded into a Pandas DataFrame, and the first few rows along with its shape are examined.
  
- **Bar Plots and Histograms:**  
  - A bar plot of `Machine failure` shows the overwhelming number of non‑failure cases.
  - Separate bar charts for each failure type (TWF, HDF, PWF, OSF, RNF) are generated.
  - For failed machines, an additional category "No Specific Failure" is used to identify cases where a failure occurred but no type was flagged.

#### 4.1.2 Observations on Class Imbalance

- **Findings:**  
  - Non‑failure instances constitute the majority of the data.
  - Among the failure types, some classes (e.g., TWF, RNF) are extremely underrepresented.
  
- **Implications:**  
  This imbalance will likely cause the model to favor the majority class, reducing sensitivity (recall) for rare failures.

### 4.2 Data Preprocessing

#### 4.2.1 Feature Selection and Encoding

- **Selected Features:**  
  Key sensor measurements such as "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", and the encoded "Type" variable are used.
  
- **Encoding:**  
  The categorical feature `Type` is transformed into numerical values using label encoding.

#### 4.2.2 Data Normalization and Splitting

- **Normalization:**  
  StandardScaler is applied to ensure that all features contribute equally during training.
  
- **Train-Test Split:**  
  The dataset is split into training (80%) and testing (20%) sets.

### 4.3 Initial Modeling Without Balancing

#### 4.3.1 Model Architecture

- **Dual‑Output Model:**  
  A deep neural network is constructed using the Keras Functional API with two outputs:
  - **Output 1:** Sigmoid activation for binary classification of `Machine failure`.
  - **Output 2:** Sigmoid (or softmax) activation for predicting multiple failure type flags.

#### 4.3.2 Training Setup and Hyperparameters

- **Optimizer:** Adam (learning rate 0.001).
- **Loss Function:** Binary crossentropy for both outputs.
- **Epochs & Batch Size:** 50 epochs with a batch size of 32.
  
#### 4.3.3 Baseline Performance Analysis

- **Results:**  
  Despite high overall accuracy, the model exhibits low recall for failure cases because of class imbalance.
  
- **Conclusion:**  
  The baseline model underscores the need for balancing the dataset to improve prediction for minority classes.

### 4.4 Modeling With Balancing Techniques

#### 4.4.1 Identification of Class Imbalance

- **Issue:**  
  The dataset is highly imbalanced with very few machine failure examples and some extremely rare failure types.

#### 4.4.2 Resampling Strategy: SMOTE

- **Implementation:**  
  SMOTE (Synthetic Minority Over-sampling Technique) is used on the training set to generate synthetic samples for underrepresented failure types.
  
- **Rationale:**  
  SMOTE creates a more balanced distribution and enables the model to learn meaningful features for minority classes.

#### 4.4.3 Aggregation of Failure Types

- **New Label – `Failure_Type`:**  
  Instead of multiple binary columns, failure type information is aggregated into a single categorical column using a priority order.
  
- **One‑Hot Encoding:**  
  The aggregated `Failure_Type` is one‑hot encoded for multi‑class classification.

#### 4.4.4 Separate Binary and Multi‑Class Models

- **Binary Model:**  
  A dedicated model is built to predict "No Machine failure" (binary output) using a sigmoid activation.
  
- **Multi‑Class Model:**  
  A separate model is developed to classify the specific failure type using a softmax activation.
  
- **Justification:**  
  This separation allows each model to specialize, leading to better overall performance and clearer interpretation of results.

### 4.5 Model Architecture Justification

- **Layer Design:**  
  Both models use several dense layers with Batch Normalization and Dropout:
  - **Dense Layers:** For feature extraction.
  - **Batch Normalization:** To stabilize and speed up training.
  - **Dropout:** To mitigate overfitting.
  
- **Activation Functions:**  
  - **ReLU:** For non‑linear transformations in hidden layers.
  - **Sigmoid/Softmax:** For probability estimation in the output layers.
  
- **Overall Rationale:**  
  The network is deep enough to capture complex patterns while remaining regularized. The SMOTE-based resampling and dual‑model approach effectively address class imbalance.

---

## 5. Training, Evaluation, and Results

### 5.1 Training Curves and Convergence

- **Observation:**  
  Both binary and multi‑class models show a smooth decrease in loss and an increase in accuracy with a minimal gap between training and validation—indicating minimal overfitting.

### 5.2 Evaluation Metrics

#### 5.2.1 Confusion Matrix and Classification Report

- **Binary Model:**  
  Confusion matrices and reports indicate high accuracy on non‑failures but lower recall on failures.
  
- **Multi‑Class Model:**  
  Performance metrics vary by failure type; common types (e.g., HDF, OSF) show high recall, while extremely rare types (e.g., TWF) remain challenging.

#### 5.2.2 ROC Curves and AUC Analysis

- **Binary ROC:**  
  An AUC near 0.98 demonstrates excellent discrimination.
  
- **Multi‑Class ROC:**  
  Most classes achieve high AUC values; however, rare classes may exhibit lower AUC.

#### 5.2.3 Feature Importance Analysis

- **Method:**  
  The average absolute weights of the first dense layer are used to assess feature importance.
  
- **Findings:**  
  Features such as "Rotational speed [rpm]", "Air temperature [K]", and "Torque [Nm]" are the most influential.

### 5.3 Observations and Discussion

- **Without Balancing:**  
  The model achieved high overall accuracy but failed to reliably detect rare failures.
  
- **With SMOTE:**  
  Balancing improved recall for minority classes. However, challenges persist for extremely rare failures (e.g., TWF).
  
- **Conclusion:**  
  While SMOTE greatly enhances performance, additional resampling strategies or advanced feature engineering may be necessary to improve detection for the rarest failure types.

---

## 6. Embedded AI Deployment

### 6.1 Model Conversion and Export

After training the multi‑class model in Colab, the model was converted to TensorFlow Lite format for deployment on the STM32L4R9 board. For example:

```python
# Convert the Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_multiclass)
tflite_model = converter.convert()

# Save the TFLite model
with open('balanced_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Export test arrays (converted to float32)
X_test_scaled_float32 = X_test_scaled.astype(np.float32)
Y_test_float32 = y_test_type_encoded.astype(np.float32)
np.save("X_test_scaled.npy", X_test_scaled_float32)
np.save("Y_test.npy", Y_test_float32)
```

### 6.2 Integration on STM32L4R9

#### 6.2.1 C Code Modifications

To deploy the model on the STM32 board, the following modifications were made:
- **STM32CubeIDE Adjustments:**  
  Removed motion SD-related code from `\IAEmbarquee\Core\Src\main.c` to streamline the application.
- **Application Code:**  
  Added custom C code in `\IAEmbarquee\X-CUBE-AI\App\app_x-cube-ai.c` to acquire data via UART, reconstruct floats from received bytes, run inference using the AI engine, and post‑process the outputs for UART transmission.

A representative snippet from the C code is:

```c
int acquire_and_process_data(ai_i8 *data[]) {
    unsigned char tmp[BYTES_IN_FLOATS] = {0};
    int num_elements = sizeof(tmp) / sizeof(tmp[0]);
    int num_floats = num_elements / 4;
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t *)tmp, sizeof(tmp), TIMEOUT);
    if (status != HAL_OK) {
      printf("Failed to receive data from UART. Error code: %d\n", status);
      return (1);
    }
    if (num_elements % 4 != 0) {
      printf("The array length is not a multiple of 4 bytes.\n");
      return (1);
    }
    for (size_t i = 0; i < num_floats; i++) {
      unsigned char bytes[4] = {0};
      for (size_t j = 0; j < 4; j++) {
        bytes[j] = tmp[i * 4 + j];
      }
      for (size_t k = 0; k < 4; k++) {
        ((uint8_t *)data)[(i * 4 + k)] = bytes[k];
      }
    }
    return (0);
}
```

#### 6.2.2 Communication via UART (Python Script)

A separate Python script was developed to send test inputs to the STM32 board and receive the predictions via UART. Key functions include synchronizing the UART, sending the input data (converted to bytes), and reading the 5‑byte output representing class probabilities:

```python
def synchronise_UART(serial_port):
    while True:
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if ret == b"\xCD":
            serial_port.read(1)
            break

def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)

def read_output_from_STM32(serial_port):
    output = serial_port.read(5)
    if len(output) < 5:
        print("Warning: Incomplete data received from STM32.")
        return []
    return [int(b)/255 for b in output]
```

The script then evaluates the model by comparing the predicted output with the expected result.

### 6.3 Encountered Challenges

During the evaluation phase using the Python communication script, the following issue was encountered:

- **Empty Data Reception:**  
  The STM32 board often did not return any data (i.e. the output from `serial_port.read(5)` was empty), leading to an error when attempting to use `np.argmax` on an empty sequence.

- **Board Compatibility:**  
  It appears that the provided board is not fully compatible with our application. The limited processing power and potential UART communication delays may prevent reliable real‑time inference, or there may be configuration issues with the board’s firmware.

- **Impact on Project:**  
  While the model was successfully converted and flashed onto the STM32L4R9, the inference phase did not operate as expected. Future work should investigate alternate communication protocols, firmware optimizations, or more capable hardware for embedded inference.

---

## 7. Conclusions and Future Work

- **Conclusions:**
  - The Colab‑based model development demonstrated that SMOTE‑based balancing and a dual‑model approach can improve predictive maintenance performance.
  - The model achieved high overall accuracy, though rare failure types (e.g., TWF) remain challenging.
  - Embedded deployment via TFLite was successfully completed, but issues in real‑time communication and board compatibility were encountered.

- **Limitations:**
  - Extremely rare failure types remain difficult to predict accurately.
  - The STM32 board showed limitations in reliably processing and transmitting inference results.

- **Future Work:**
  - Explore alternative resampling methods or feature engineering to further improve predictions for rare failure types.
  - Optimize the UART communication and firmware on the embedded device.
  - Evaluate the deployment on more powerful embedded platforms or consider hybrid architectures.

---

## 8. Instructions for Execution

1. **Environment Setup:**  
   Install required Python packages:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn imbalanced-learn pyserial
   ```
2. **Data Access:**  
   Place `ai4i2020.csv` in an accessible location (e.g., mount Google Drive in Colab) and update file paths in the code accordingly.
3. **Colab Workflow:**
   - Load and preprocess the data.
   - Train and evaluate the models (both without and with SMOTE).
   - Convert the trained multi‑class model to TFLite and export test datasets.
4. **Embedded Deployment:**
   - Open the provided STM32CubeIDE project and apply the C code modifications.
   - Flash the generated TFLite model onto the STM32 board.
   - Run the Python communication script (e.g., `Communication_STM32_NN.py`) to evaluate model inference on the board.
5. **Troubleshooting:**
   - If UART communication fails (empty outputs), check hardware connections and firmware configuration.
   - Adjust UART timeout and synchronization parameters as needed.

---

## 9. References

- [Scikit-learn: Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Scikit-learn: ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
- [Imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [TensorFlow Tutorial on Imbalanced Data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=fr)
- [STM32Cube.AI Documentation](https://www.st.com/en/embedded-software/stm32cube-ai.html)

---

## 10. Authors and License

**Author:** Yasser EL KOUHEN  
**Supervision:** Professor Kévin HECTOR, Mines Saint-Étienne  

