# 🧬 AI-Driven Drug Discovery: COX Inhibitor Classification

### 📋 Project Overview
This project focuses on **Computational Drug Discovery**, specifically aiming to classify chemical compounds based on their biological activity against **Cyclooxygenase (COX-1 / COX-2)** enzymes.

Using a dataset of chemical descriptors derived from ChEMBL, I developed and compared multiple Machine Learning models to predict whether a molecule is active or inactive. The pipeline includes extensive data preprocessing, handling class imbalance, and hyperparameter tuning to optimize for high Recall and F1-scores, ensuring potential drug candidates are not missed during screening.

### 🛠️ Tech Stack & Tools
* **Language:** Python 3.x
* **Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
* **Techniques:** Supervised Learning, Data Balancing, Grid Search CV

### 📊 Dataset & Features
The dataset (`cox1_cox2_good.csv`) consists of chemical compounds represented by various molecular descriptors, including:
* **Electronic properties:** MaxPartialCharge, MinAbsPartialCharge, etc.
* **Topology/Connectivity:** BCUT2D metrics (MWHI, LOGPHI, MRHI, etc.).

**Preprocessing Pipeline:**
1.  **Cleaning:** Removal of identifiers (`ChEMBL ID`, `Smiles`) to focus purely on physicochemical properties.
2.  **Imputation:** Handling missing values using statistical mean strategies.
3.  **Balancing:** Addressing dataset imbalance through undersampling to ensure the model learns to identify both classes effectively (Train/Val/Test split: 70/15/15).
4.  **Scaling:** Applied `StandardScaler` to normalize feature distributions.

### 🤖 Models & Evaluation
I rigorously tested 13 different classification algorithms to find the optimal architecture:

* **Tree-based:** Decision Tree, Random Forest, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost.
* **Linear/Probabilistic:** Logistic Regression, Gaussian Naive Bayes, QDA.
* **Other:** SVM (Linear & RBF), k-Nearest Neighbors (KNN), MLP Classifier (Neural Network).

**Performance Metrics:**
Given the nature of drug discovery, specific focus was placed on:
* **Recall:** To minimize False Negatives (missing a potential drug).
* **Precision & F1-Score:** To ensure the reliability of predictions.
* **Confusion Matrices:** To visualize True Positives vs False Positives.

### 🏆 Results Highlight
* **Grid Search** was utilized to tune hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`).
* Models like **HistGradientBoosting** and **SVC** demonstrated strong performance after tuning, achieving high recall rates on the validation set.

### 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/TasosTzaras/cox-inhibitor-prediction.git](https://github.com/TasosTzaras/cox-inhibitor-prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook final_code.ipynb
    ```

### 👨‍💻 Author
**Tasos Tzaras**
* Software Engineer | AI & Drug Discovery Enthusiast
* [LinkedIn Profile](https://www.linkedin.com/in/tasos-tzaras)
* [Portfolio Website](https://tasostzaras.github.io/tasostzaras/)

---
*This project was developed as part of my Master's Thesis research at the University of Ioannina.*
