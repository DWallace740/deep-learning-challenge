# Alphabet Soup Funding Success Prediction
**This README serves as both the project documentation and the final report for the Deep Learning Challenge.**  
It contains all required sections, including an overview, results, model evaluation, and recommendations.

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

**Optimizations Applied:**
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
- Python
- Pandas for data manipulation
- Scikit-Learn for preprocessing and scaling
- TensorFlow and Keras for deep learning modeling
- Jupyter Notebook for development

## Repository Structure
deep-learning-challenge/ │── AlphabetSoupCharity.ipynb # Initial model notebook │── AlphabetSoupCharity_Optimization.ipynb # Optimized model notebook │── AlphabetSoupCharity.h5 # Saved initial model │── AlphabetSoupCharity_Optimization.h5 # Saved optimized model │── README.md # 

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/deep-learning-challenge.git

2. Navigate to the project directory:
    ```bash
    cd deep-learning-challenge

3. Install Dependencies:
    ```bash 
    pip install -r requirements.txt

4. Run the Jupyter Notebook: 
    ```bash 
    jupyter notebook AlphabetSoupCharity_Optimization.ipynb

## Resources and Support
This project was completed using a combination of coursework, instructor resources, and external tools. 

# Below are the key resources used:
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-Learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Keras Sequential API Guide: https://keras.io/guides/sequential_model/

# Support:
- ChatGPT (AI Assistant): Provided guidance on code logic, debugging, structuring scripts, and formatting this README file.
- Xpert Learning: Consulted for guidance on neural network architectures, hyperparameter tuning, and best practices for deep learning model optimization.
- Class Materials: Course slides and lectures on deep learning and classification models.

All external resources used are listed here for transparency.

## Author
Daena Wallace