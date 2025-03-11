# Deep Learning Challenge: Alphabet Soup Funding Success Prediction

## Overview of the Analysis
This project focuses on building a machine learning model to predict whether an organization applying for funding from Alphabet Soup will successfully utilize the funds. The dataset consists of historical funding applications, with various organizational characteristics as input features. A deep learning model using TensorFlow and Keras was developed to classify applications as successful or unsuccessful.

## Results

### 1. Data Preprocessing
- **Target Variable:** `IS_SUCCESSFUL` (1 = Successful, 0 = Unsuccessful)
- **Feature Variables:** All columns except `EIN` and `NAME`
- **Removed Variables:**  
  - `EIN`: A unique identifier with no predictive value  
  - `NAME`: Non-informative for classification  
- **Preprocessing Steps:**  
  - Categorical variables were one-hot encoded using `pd.get_dummies()`.  
  - The dataset was split into **training (80%) and testing (20%)** sets.  
  - Features were scaled using `StandardScaler()`.

### 2. Model Compilation, Training, and Evaluation

#### Initial Model
| Layer | Neurons | Activation Function |
|--------|---------|----------------------|
| **Input Layer** | Number of features | - |
| **Hidden Layer 1** | 80 | ReLU |
| **Hidden Layer 2** | 30 | ReLU |
| **Output Layer** | 1 | Sigmoid |

- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 100  

#### Initial Model Performance
| Dataset | Accuracy |
|---------|----------|
| Training | 73.5% |
| Testing  | 72.0% |

The initial model failed to reach the target accuracy of 75%, requiring optimization.

### 3. Model Optimization

#### Optimized Model
| Layer | Neurons | Activation Function |
|--------|---------|----------------------|
| **Input Layer** | Number of features | - |
| **Hidden Layer 1** | 100 | ReLU |
| **Hidden Layer 2** | 50 | ReLU |
| **Hidden Layer 3** | 25 | ReLU |
| **Output Layer** | 1 | Sigmoid |

#### Optimizations Applied:
1. Increased neurons in the first and second hidden layers.
2. Added a third hidden layer.
3. Increased training epochs from 100 to 150.

#### Optimized Model Performance
| Dataset | Accuracy |
|---------|----------|
| Training | 75.8% |
| Testing  | 75.1% |

The optimized model exceeded the 75% accuracy target and was saved as:
- `AlphabetSoupCharity.h5` (Initial Model)
- `AlphabetSoupCharity_Optimization.h5` (Optimized Model)

## Summary and Recommendations

### Strengths:
- The optimized deep learning model reached the target accuracy of 75%.
- The model effectively classifies successful and unsuccessful funding applications.
- Feature scaling and one-hot encoding improved model performance.

### Weaknesses:
- The dataset consists primarily of structured categorical data, which may not be ideal for deep learning.
- The model’s accuracy could still be improved with further feature engineering.

### Alternative Models:
A deep learning approach may not be the best fit for structured tabular data. Alternative models that could perform better include:
1. **Random Forest Classifier** – Handles categorical data well and prevents overfitting.
2. **Gradient Boosting (XGBoost)** – More interpretable and often outperforms deep learning on structured data.
3. **Logistic Regression** – Simple and effective for binary classification.

Future improvements could include:
- Feature selection to reduce dimensionality.
- Hyperparameter tuning of the deep learning model.
- Testing alternative machine learning models.

## Technologies Used
- **Python**
- **Pandas** for data manipulation
- **Scikit-Learn** for preprocessing and scaling
- **TensorFlow and Keras** for deep learning modeling
- **Jupyter Notebook** for development

## Author
**Daena Wallace**