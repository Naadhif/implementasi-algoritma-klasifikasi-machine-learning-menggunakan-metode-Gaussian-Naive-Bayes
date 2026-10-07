# Retail Product Classification using Gaussian Naive Bayes

A machine learning project implementing the **Gaussian Naive Bayes** algorithm to predict and classify retail product categories based on features such as price, rating, stock status, and discount.

---

## 📌 Project Overview
This repository contains a complete Python workflow demonstrating foundational data science practices—from exploratory data analysis and data preprocessing to model training, evaluation, and prediction using `scikit-learn`.

---

## 🚀 Features & Workflow
1. **Data Loading & Inspection**: 
   - Imports datasets (`retail_product_dataset_300.csv`).
   - Checks data dimensions, missing values, and data types.
2. **Data Preprocessing**:
   - Handles missing values via median imputation (`SimpleImputer`).
   - Encodes categorical stock status into dummy/indicator variables (`pd.get_dummies`).
3. **Data Splitting**:
   - Splits data into training (80%) and testing (20%) sets using stratified sampling to maintain class balance.
4. **Model Training (Gaussian Naive Bayes)**:
   - Fits a `GaussianNB` classifier on the training data.
   - Inspects prior probabilities and learned statistical parameters (mean and variance per feature per class).
5. **Evaluation & Prediction**:
   - Generates class probabilities and predictions on unseen test data.

---

## 🗂️ Dataset Structure
The dataset (`retail_product_dataset_300.csv`) consists of 300 rows and 5 primary columns:
- `category`: Target class (e.g., *Beauty*, *Clothing*, *Electronics*, *Home & Kitchen*, *Sports*).
- `price`: Numerical product price.
- `rating`: Customer rating.
- `stock`: Categorical stock availability status.
- `discount`: Applicable discount percentage.

---

## 🛠️ Requirements & Installation
Make sure you have the following Python libraries installed before running the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 💻 Usage
Clone this repository and open the Jupyter Notebook to explore the step-by-step implementation:

```bash
git clone https://github.com/username/retail-naive-bayes.git
cd retail-naive-bayes
jupyter notebook retail_product_classification.ipynb
```

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
