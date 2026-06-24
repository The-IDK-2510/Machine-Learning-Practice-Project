# Comment Category Prediction Challenge (College Course Project)

## Project Overview
This repository documents my first implementation of a machine learning workflow, developed as a final project for a college course using a private Kaggle competition environment. The objective of the challenge was to explore an unstructured discussion dataset and build a predictive classifier capable of automatically mapping user comments to four unique system categories.

The primary codebase for this project is contained within the verbatim script file: `24f2005843-notebook-t12026.ipynb`.

**Final Performance:** 
* **Evaluation Metric:** Test F1 Score
* **Achieved Score:** 0.80146
* **Kaggle Leaderboard Rank:** 1793 out of 2744 competitors

> **Data Privacy Note:** The underlying competition data is proprietary academic material and cannot be uploaded or shared publicly. This repository serves as a code-only documentation showcase of the logic and methods used.

---

## Technical Workflow & Methodology

As my introductory machine learning project, the modeling tasks were performed sequentially to clean the dataset, transform categorical variables, vectorize text blocks, and evaluate algorithmic execution.

### 1. Exploratory Data Analysis & Visualization
* Analyzed the count distribution of the 4 unique target labels to diagnose severe class imbalances.
* Plotted correlation matrices for numerical metadata features (such as upvotes, downvotes, and system emoticons) to extract behavioral patterns from the short textual data.

### 2. Step-by-Step Data Preprocessing
* **Imputation & Encoding:** Managed missing demographic labels and internal flags by applying explicit data input strategies via `SimpleImputer` and converting text categories into numerical arrays using `OneHotEncoder`.
* **Text Representation:** Applied a `TfidfVectorizer` to tokenize, scale, and evaluate term importance within the raw text blocks.
* **Matrix Stacking:** Manually merged sparse text metrics with independent metadata arrays horizontally (`hstack`) to generate a memory-efficient `csr_matrix` array optimized for model inputs.

### 3. Algorithm Training & Validation
Using a train-test split configuration, multiple distinct classifier architectures were implemented and tested to isolate performance benchmarks:
* **Baseline Baseline:** `DummyClassifier`
* **Text Classifiers:** Multinomial Naive Bayes (`MultinomialNB`) & Linear Support Vector Machines (`LinearSVC`)
* **Tree Ensembles:** `RandomForestClassifier`
* **Boosting Frameworks:** Gradient Boosting structures including `XGBClassifier` and `LGBMClassifier`

Final configurations were refined using standalone scoring validations to prevent algorithmic overfitting.

---

## Technologies Used
* **Languages:** Python
* **Libraries:** Scikit-learn, LightGBM, XGBoost, Pandas, NumPy, Scipy, Matplotlib, Seaborn
* **Environment:** Kaggle Notebooks
