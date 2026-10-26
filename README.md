[README.md](https://github.com/user-attachments/files/33962040/README.md)
# 🎓 Student Placement & Salary Prediction

A machine learning project that explores student academic, technical, and experience-related data to predict **placement status** and estimate **salary package (LPA)**. The project is developed in a Jupyter Notebook and includes exploratory data analysis (EDA), data preprocessing, classification, regression, model evaluation, hyperparameter tuning, and saving model artifacts for a Streamlit application.

> **Note:** Results below are copied from the outputs saved in the supplied notebook. Your results may differ if you rerun the notebook with different data, library versions, or code changes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Tasks](#machine-learning-tasks)
- [Workflow](#workflow)
- [Models Used](#models-used)
- [Evaluation Results](#evaluation-results)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Project Files](#project-files)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Artifacts](#model-artifacts)
- [Important Implementation Notes](#important-implementation-notes)
- [Future Improvements](#future-improvements)

## Project Overview

The purpose of this project is to investigate how student characteristics relate to placement outcomes and salary packages. It contains two predictive tasks:

1. **Classification:** predict whether a student is `Placed` or `Not Placed`.
2. **Regression:** estimate a placed student's salary package in **Lakhs Per Annum (LPA)**.

The notebook also demonstrates common machine learning practices such as exploratory data analysis, encoding categorical features, scaling numerical data, cross-validation, regularization, model comparison, and hyperparameter tuning.

## Dataset

The notebook loads the dataset from:

```text
student_placement_prediction_dataset_2026.csv
```

The dataset inspected in the notebook contains **100,000 rows and 25 columns**. The `student_id` column is removed before analysis. The two target columns are:

- `placement_status` — classification target.
- `salary_package_lpa` — regression target.

### Features in the dataset

| Feature | Description |
|---|---|
| `age` | Student age |
| `gender` | Gender category |
| `cgpa` | Cumulative grade point average |
| `branch` | Academic branch |
| `college_tier` | College tier/category |
| `internships_count` | Number of internships |
| `projects_count` | Number of projects |
| `certifications_count` | Number of certifications |
| `coding_skill_score` | Coding skill score |
| `aptitude_score` | Aptitude score |
| `communication_skill_score` | Communication skill score |
| `logical_reasoning_score` | Logical reasoning score |
| `hackathons_participated` | Number of hackathons participated in |
| `github_repos` | Number of GitHub repositories |
| `linkedin_connections` | Number of LinkedIn connections |
| `mock_interview_score` | Mock interview score |
| `attendance_percentage` | Attendance percentage |
| `backlogs` | Number of backlogs |
| `extracurricular_score` | Extracurricular activity score |
| `leadership_score` | Leadership score |
| `volunteer_experience` | Volunteer experience category |
| `sleep_hours` | Hours of sleep |
| `study_hours_per_day` | Daily study hours |
| `placement_status` | Placement outcome (`Placed` / `Not Placed`) |
| `salary_package_lpa` | Salary package in LPA |

The notebook reports **no missing values and no duplicate rows** in the inspected dataset. The placement target distribution is:

- `Placed`: 54,459 records
- `Not Placed`: 45,541 records

**Dataset availability:** place `student_placement_prediction_dataset_2026.csv` in the same directory as the notebook before running it. If the dataset is not included in this GitHub repository, you must add it separately or update the data-loading path. Do not assume that uploading the notebook also uploads the CSV dataset.

## Machine Learning Tasks

### 1. Placement Classification

The classification model predicts one of two classes:

- `Placed`
- `Not Placed`

The notebook removes both `placement_status` and `salary_package_lpa` from the input features for this task. It uses a stratified 80/20 train-test split, then trains and evaluates multiple classifiers.

### 2. Salary Regression

The regression task predicts `salary_package_lpa`. The notebook trains regression models using records whose salary package is greater than zero, representing placed students with a positive salary value.

Regression performance is assessed using:

- **R² Score:** how much variance in the target is explained by the model; closer to 1 is generally better.
- **MAE:** mean absolute prediction error, reported in LPA.
- **RMSE:** root mean squared prediction error, reported in LPA; it penalizes larger errors more strongly than MAE.

## Workflow

### 1. Environment and setup

The notebook imports data processing, visualization, model training, evaluation, serialization, and utility libraries. It sets a fixed random seed (`42`) and creates an artifact directory.

### 2. Data inspection and exploratory data analysis (EDA)

The notebook performs:

- Dataset preview and data-type inspection with `head()` and `info()`.
- Removal of `student_id`.
- Missing-value checks.
- Duplicate-row checks.
- Descriptive statistics using `describe()`.
- Placement class-distribution visualization.
- Feature-distribution histograms for selected variables.
- Boxplots comparing selected features by placement status.
- A correlation heatmap.
- Outlier counts using the Interquartile Range (IQR) method.

These steps help inspect the dataset and understand feature distributions before modeling.

### 3. Train-test split

The data is split into **80% training data and 20% testing data**. The classification split uses `stratify=y` so the proportions of `Placed` and `Not Placed` remain similar in both sets. The notebook output shows 80,000 training rows and 20,000 test rows.

### 4. Categorical encoding

The notebook uses two encoding approaches:

- **Ordinal Encoding:** applied to `college_tier` and `branch`.
- **One-Hot Encoding:** applied to `gender` and `volunteer_experience`, with `drop="first"` and `handle_unknown="ignore"`.

Encoding converts categorical values into numeric representations that machine learning models can process.

### 5. Feature scaling

The notebook demonstrates two scaling methods:

- **Min-Max Scaling (`MinMaxScaler`):** transforms features to a specified range, by default 0 to 1.
- **Standardization (`StandardScaler`):** transforms features based on the training-set mean and standard deviation, producing approximately zero mean and unit variance for standardized training features.

The notebook displays before-and-after comparisons for selected columns.

### 6. Cross-validation and regularization

The notebook uses 5-fold cross-validation to estimate classification performance across different data folds. It also compares Logistic Regression with L1 and L2 regularization.

- **L1 regularization** can shrink some model coefficients to zero.
- **L2 regularization** shrinks coefficient values toward zero.
- In Logistic Regression, `C` is the inverse of regularization strength: a smaller `C` means stronger regularization.

### 7. Classification model training

The notebook trains five classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

The classifiers are configured with class weighting where shown in the notebook to help account for the class distribution.

### 8. Regression model training

The notebook trains three regression algorithms on records with a positive salary package:

1. Linear Regression
2. Gradient Boosting Regressor
3. Random Forest Regressor

The models are compared using R², MAE, and RMSE.

### 9. Classification evaluation

The notebook evaluates classification models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- ROC curve and ROC-AUC

These metrics describe different aspects of model performance. In particular, precision measures how often positive predictions are correct, while recall measures how many actual positive cases the model identifies.

### 10. Hyperparameter tuning

The notebook uses `GridSearchCV` with 5-fold cross-validation to tune Logistic Regression. It searches combinations of:

- `C`: `0.01`, `0.1`, `1`, `10`, `100`
- `penalty`: `l1`, `l2`
- `solver`: `liblinear`
- `class_weight`: `balanced`

The tuning scorer is the F1-score for the `Placed` class.

### 11. Saving models and schemas

The notebook saves trained models with `joblib` and feature metadata with JSON. The intended artifacts include:

- Classification model
- Classification scaler
- Classification schema
- Regression model
- Regression schema

The schema describes the task, model name, target, and feature information for use by a dynamic Streamlit interface.

### 12. Streamlit application

The notebook includes a cell that writes a `streamlit_app.py` file. The app is intended to load a saved model and its JSON schema, collect student details through a web interface, and display either a placement prediction or an estimated salary package.

## Models Used

| Task | Algorithm | Purpose |
|---|---|---|
| Classification | Logistic Regression | Predict placement class |
| Classification | Decision Tree | Predict placement class |
| Classification | Random Forest Classifier | Predict placement class |
| Classification | K-Nearest Neighbors | Predict placement class |
| Classification | Support Vector Machine | Predict placement class |
| Regression | Linear Regression | Estimate salary package |
| Regression | Gradient Boosting Regressor | Estimate salary package |
| Regression | Random Forest Regressor | Estimate salary package |

## Evaluation Results

### Classification — tuned Logistic Regression

The notebook's recorded test-set results for the tuned Logistic Regression model are:

| Metric | Recorded result |
|---|---:|
| Accuracy | 0.5576 |
| Precision (`Placed`) | 0.5989 |
| Recall (`Placed`) | 0.5684 |
| F1-score (`Placed`) | 0.5832 |
| ROC-AUC | 0.5854 |
| Best cross-validation F1-score | 0.5868 |

The recorded best hyperparameters are:

```text
C = 0.1
class_weight = balanced
penalty = l2
solver = liblinear
```

The notebook also records a mean 5-fold accuracy of **0.5695** for its initial Logistic Regression cross-validation experiment.

### Regression — comparison on positive-salary records

| Model | R² Score | MAE (LPA) | RMSE (LPA) |
|---|---:|---:|---:|
| Linear Regression | 0.6028 | 0.7958 | 0.9985 |
| Gradient Boosting | 0.5992 | 0.7993 | 1.0030 |
| Random Forest | 0.5905 | 0.8076 | 1.0138 |

Based on the recorded R² ranking, **Linear Regression** is the best of these three regression models in this notebook run.

These are notebook-recorded results, not a guarantee of performance on new or real-world student data. The classification results are modest, so additional validation and improvements are recommended before practical use.

## Libraries and Dependencies

### Third-party Python packages

| Package | Used for |
|---|---|
| `numpy` | Numerical operations and array calculations |
| `pandas` | Loading, cleaning, transforming, and analyzing tabular data |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Statistical plots, including histograms, boxplots, count plots, and heatmaps |
| `scikit-learn` | Preprocessing, train-test splitting, cross-validation, classification, regression, metrics, and hyperparameter tuning |
| `joblib` | Saving and loading trained machine learning models and scalers |
| `streamlit` | Building the interactive prediction web application |

### Python standard-library modules

These are imported by the notebook but do **not** need to be installed separately:

- `json` — reading and writing JSON schemas.
- `os` — filesystem and directory operations.
- `warnings` — controlling warning output.
- `time` — measuring model-training time.

The notebook also uses Jupyter/IPython notebook functionality such as `display()` and the `%%writefile` cell magic. Run the notebook in Jupyter Notebook, JupyterLab, or a compatible notebook environment. If you run only the app, you do not need the notebook magic.

### Install the required packages

Create a `requirements.txt` file with:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

Then install the packages:

```bash
python -m pip install -r requirements.txt
```

For notebook execution, install Jupyter if it is not already available:

```bash
python -m pip install notebook
```

No separate installation is needed for `json`, `os`, `warnings`, or `time`. The notebook does not show a requirement for a paid API or external AI service.

## Project Files

A typical repository layout for the project is:

```text
ML-Project/
├── ML project.ipynb
├── student_placement_prediction_dataset_2026.csv
├── streamlit_app.py
├── requirements.txt
├── README.md
└── MODEL_FILES/
    ├── classification_model.pkl
    ├── classification_scaler.pkl
    ├── classification_schema.json
    ├── regression_model.pkl
    └── regression_schema.json
```

Your actual repository may contain a different subset of these files. The notebook generates model artifacts when the relevant cells run; `streamlit_app.py` is written by a notebook cell in the supplied notebook. If the dataset or artifacts are not present in GitHub, add them or generate them before following the run instructions.

## Installation

1. Install Python.
2. Clone or download this repository.
3. Open a terminal in the project folder.
4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Ensure the CSV dataset is in the project folder.
6. Open `ML project.ipynb` in Jupyter Notebook or JupyterLab.

## How to Run

### Run the notebook

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Open `ML project.ipynb` and execute the cells in order. The notebook expects the CSV file to be available at the relative path shown in the data-loading cell.

### Run the Streamlit app

After confirming that `streamlit_app.py` and the expected model files are present, run:

```bash
streamlit run streamlit_app.py
```

The app expects the model files and JSON schemas in the `MODEL_FILES/` directory. If these files have not been generated, run the relevant model-training and artifact-saving cells first.

## Important Implementation Notes

The supplied notebook contains a few implementation details that should be checked before presenting the Streamlit app as fully working:

1. **Schema key mismatch:** the artifact-saving code writes feature metadata using the key `dtype` and values such as `numeric`, while the app code checks `feat["type"]` and the value `numerical`. These keys and values need to be made consistent.
2. **Classification scaler mismatch:** the notebook trains classification models using `StandardScaler`-transformed data (`X_train_std`), but the artifact-saving code saves the variable named `scaler`, which was fitted as a `MinMaxScaler`. The saved scaler should match the exact preprocessing used to train the saved classification model.
3. **Artifact directory:** the notebook creates `model_artifacts` during setup, but its later artifact-saving code and Streamlit app use `MODEL_FILES`. Use one consistent directory.
4. **Regression feature consistency:** the regression model is trained on the encoded, unscaled feature data. The app's regression inputs and feature schema must use exactly the same feature columns, ordering, encoding, and preprocessing as training.
5. **Target and schema labels:** the classification schema saves `class_names`, while the app checks for `target_classes` when mapping prediction labels. Align these keys if that mapping is needed.
6. **App input types:** the schema-building function currently generates numeric columns only. If categorical inputs are needed in the app, their options and encoding must be included in the schema and the inference pipeline.

These are observations from reading the supplied notebook code; they are not fixes applied to the project. Check and test these items before relying on predictions from the web app.

## Future Improvements

- Build preprocessing and model training into a consistent scikit-learn `Pipeline` or `ColumnTransformer`.
- Fit encoders and scalers only on training data, then reuse the exact fitted transformations during inference.
- Compare more models and tune the decision threshold based on the intended use of the placement prediction.
- Investigate the features most associated with placement and salary, while avoiding claims of causation from correlation alone.
- Add a reproducible environment file with tested package versions.
- Add validation for user input and clear error messages in the Streamlit app.
- Document the dataset's source, license, and whether it contains synthetic or real student records.

## Disclaimer

This project is for educational and demonstration purposes. Predictions are estimates based on the supplied dataset and trained models. They should not be treated as guarantees of placement or salary outcomes.

## Author

Add your name, GitHub profile, and contact details here.
