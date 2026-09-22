# Personal Loan Campaign — Customer Conversion Prediction

## Project Overview

AllLife Bank wants to increase the number of asset customers by converting existing liability customers into personal-loan customers. A previous campaign produced roughly a 9% conversion rate, so the objective of this project is to build a classification model that can identify customers who are more likely to accept a personal-loan offer.

The project combines exploratory data analysis, data preprocessing, class-imbalance-aware model building, hyperparameter tuning, model comparison, and business recommendations for targeted marketing.

## Business Objective

Build a predictive model that helps the bank:

- identify customers with a high probability of accepting a personal loan;
- improve campaign conversion rates;
- reduce unnecessary marketing outreach;
- allocate marketing resources to higher-value prospects;
- grow the bank's loan portfolio and interest income.

## Dataset

The dataset contains **5,000 customer records and 14 attributes**.

Important variables include:

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Experience` | Years of professional experience |
| `Income` | Annual income in thousands of dollars |
| `Family` | Family size |
| `CCAvg` | Average monthly credit-card spending in thousands of dollars |
| `Education` | Education level |
| `Mortgage` | Mortgage value in thousands of dollars |
| `Securities_Account` | Whether the customer has a securities account |
| `CD_Account` | Whether the customer has a certificate-of-deposit account |
| `Online` | Whether the customer uses internet banking |
| `CreditCard` | Whether the customer holds another bank's credit card |
| `Personal_Loan` | Target: whether the customer accepted the previous loan offer |

`ID` and `ZIPCode` are excluded from modeling because they function primarily as identifier/location-code fields rather than direct behavioral predictors.

## Project Workflow

1. **Data Overview**
   - inspected shape, data types, missing values, duplicates, and descriptive statistics;
   - examined target-class distribution;
   - identified invalid negative values in `Experience`.

2. **Exploratory Data Analysis**
   - studied mortgage distribution and outliers;
   - analyzed credit-card ownership;
   - examined correlations with personal-loan acceptance;
   - compared loan acceptance across income, credit-card spending, education, and age-related characteristics.

3. **Data Preprocessing**
   - corrected negative experience values using age-specific median experience;
   - dropped identifier columns;
   - used a stratified train/test split because the positive class is imbalanced;
   - scaled numerical variables;
   - one-hot encoded the categorical `Education` feature;
   - packaged preprocessing into a `ColumnTransformer` / `Pipeline` workflow.

4. **Model Building**
   - Logistic Regression with balanced class weights;
   - Decision Tree Classifier;
   - Random Forest Classifier.

5. **Hyperparameter Tuning**
   - Logistic Regression tuned with `GridSearchCV`;
   - Random Forest tuned with `RandomizedSearchCV`;
   - ROC-AUC used as the primary tuning metric.

6. **Final Model Selection**
   - tuned models compared using ROC-AUC, recall, F1-score, precision, and accuracy;
   - final model serialized with `joblib` for reuse.

## Model Evaluation Strategy

The positive class represents customers who accepted the loan offer and is relatively rare. Because of this imbalance, **accuracy alone is not sufficient**.

The project emphasizes:

- **ROC-AUC** — overall ranking ability across thresholds;
- **Recall** — ability to identify customers who are actually likely to accept a loan;
- **Precision** — proportion of targeted customers who are true prospects;
- **F1-score** — balance between precision and recall.

## Final Results

The **Tuned Random Forest** produced the strongest overall performance in the completed notebook.

Approximate reported test results:

| Metric | Result |
|---|---:|
| ROC-AUC | ~0.998 |
| Recall | 0.95 |
| Precision | 0.92 |
| F1-score | ~0.93 |
| Accuracy | ~0.99 |

The model identified **114 of 120** loan-accepting customers in the test set while maintaining high precision.

## Key Business Insights

- Customers with stronger income and spending characteristics show greater potential for loan conversion.
- Since the positive class is small, broad untargeted marketing would waste campaign resources.
- A probability-based targeting strategy allows the bank to prioritize customers who are more likely to respond.
- The decision threshold can be adjusted according to campaign objectives:
  - lower threshold → higher recall and broader outreach;
  - higher threshold → higher precision and lower marketing cost.

## Business Recommendations

1. Use the tuned model to **score existing liability customers** before launching a campaign.
2. Prioritize high-probability prospects rather than contacting the entire customer base.
3. Use a staged campaign strategy: lower-cost digital channels first, followed by higher-cost calls for top prospects.
4. Monitor precision, recall, conversion rate, and campaign ROI after deployment.
5. Recalibrate the probability threshold based on business priorities and outreach capacity.
6. Retrain the model periodically as customer behavior and product offerings change.

## Repository Structure

```text
personal-loan-campaign/
├── README.md
├── Personal_Loan_Campaign_Modeling.ipynb
├── Loan_Modelling.csv
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- Joblib
- Jupyter Notebook / Google Colab

## How to Run the Project

### Option 1 — Google Colab

1. Download or clone this repository.
2. Open `Personal_Loan_Campaign_Modeling.ipynb` in Google Colab.
3. Upload `Loan_Modelling.csv` to the Colab session if it is not already available.
4. Make sure the CSV and notebook are in the same working directory.
5. Run the notebook cells from top to bottom.

The notebook includes a package-installation cell. After running it, restart the Colab runtime if prompted:

```text
Runtime → Restart session
```

Then continue executing the remaining cells.

### Option 2 — Run Locally with Jupyter

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/personal-loan-campaign.git
cd personal-loan-campaign
```

Create a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Personal_Loan_Campaign_Modeling.ipynb
```

and run the cells sequentially.

If Jupyter is not already installed, install it with:

```bash
pip install notebook
```

## Generated Model Artifact

The final notebook saves the selected model as:

```text
final_loan_model.pkl
```

The artifact is excluded from Git through `.gitignore` because it can be reproduced by running the notebook.

## Notes

- Keep `Loan_Modelling.csv` in the same directory as the notebook unless you modify `DATA_PATH`.
- The notebook uses a fixed random state where appropriate to improve reproducibility.
- Model results can vary slightly across library versions or execution environments.

## Future Improvements

- Add threshold optimization based on expected campaign cost and revenue.
- Introduce SHAP-based model explainability for customer-level decisions.
- Package the final model as a REST API.
- Add experiment tracking and model-version management.
- Evaluate performance drift after real campaign results become available.

---

This project demonstrates an end-to-end machine-learning classification workflow for a real-world banking marketing problem, from exploratory analysis and preprocessing through model tuning and business decision support.
