<div align="center">

# 🏠 Robust Regression Engine
### House Price Prediction with Regularization, Cross-Validation, Trees & SVR

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**[🚀 Live App](#-links)** · **[📓 Notebook](PR2_Robust_Regression.ipynb)** · **[🎥 Video Explanation](#-links)**

</div>

---

## 📌 Overview

A real estate analytics company has a basic house-price model that **overfits** and gives **unstable predictions**. This project builds a **robust regression pipeline** that:

- applies **regularization** (Ridge / Lasso) with tuned `alpha`
- uses **proper cross-validation** (K-Fold, Stratified K-Fold, LOOCV, Time Series Split)
- compares **linear vs non-linear** models (Ridge, Lasso, Decision Tree, Random Forest, SVR)
- selects the **best model on validation performance** and serves it through a **Streamlit app**

## 📊 Dataset

| Item | Detail |
|---|---|
| File | `Advanced_Regression_HousePrice_Dataset_3800.xlsx` |
| Rows / Columns | 3,800 / 12 (no missing values) |
| Period | Jan 2010 – Dec 2023 |
| Target | `house_price_inr` (modelled in **lakh INR**) |

**Features used:** `area_sqft`, `bedrooms`, `bathrooms`, `location_score`, `property_age`, `distance_city_km`, `near_school`, `near_metro`, `crime_rate_index`, `sale_year` (derived from `sale_date`).
`property_id` is dropped (identifier only).

## 🧭 Project Workflow

```
Load data → EDA → Train/Test split (80/20) → Scale (inside Pipeline)
   → Ridge & Lasso (tune alpha) → CV strategy comparison
   → Decision Tree & Random Forest → SVR (Linear, RBF)
   → Compare all models → Pick best by CV RMSE → Save model.pkl → Streamlit app
```

| Part | Topic | What was done |
|---|---|---|
| A | Theory | Regularization, L1 vs L2, CV types, trees vs scaling |
| B | Data prep | Features/target, 80/20 split, `StandardScaler` in a pipeline (no leakage) |
| C | Ridge & Lasso | `alpha` tuned with 5-fold GridSearchCV, coefficient comparison |
| D | CV strategies | K-Fold, Stratified (binned target), LOOCV, Time Series Split |
| E | Trees | Decision Tree (depth / min-leaf control), Random Forest |
| F | SVR | Linear and RBF kernels, tuned `C`, `gamma`, `epsilon` |
| G | Evaluation | MSE, MAE, RMSE, R², over/underfitting diagnosis |
| H | Final analysis | Best model, impact of regularization, business interpretation |

## 🏆 Results

All scores are on the **held-out test set** (RMSE / MAE in lakh INR).

| Model | CV RMSE | Train R² | Test R² | Test RMSE | Test MAE | Diagnosis |
|---|---|---|---|---|---|---|
| Ridge | 24.95 | 0.917 | 0.920 | 25.40 | 19.45 | Good fit |
| Lasso | 24.95 | 0.917 | 0.920 | 25.42 | 19.46 | Good fit |
| Decision Tree | 27.35 | 0.926 | 0.907 | 27.43 | 20.36 | Good fit |
| Random Forest | 23.78 | 0.965 | 0.929 | 23.95 | 17.50 | Good fit (slight gap) |
| SVR (Linear) | 25.07 | 0.916 | 0.918 | 25.67 | 19.37 | Good fit |
| **SVR (RBF)** ⭐ | **22.25** | 0.938 | **0.939** | **22.20** | **16.23** | **Good fit** |

**Best hyperparameters:** Ridge α = 2.04 · Lasso α = 0.117 · Tree `max_depth=7, min_samples_leaf=20` · Random Forest `max_depth=12, min_samples_leaf=5` · SVR-RBF `C=300, gamma=0.01, epsilon=0.1`.

### Cross-validation comparison (tuned Ridge)

| Strategy | Folds | Mean RMSE |
|---|---|---|
| K-Fold | 5 | 24.93 |
| Stratified K-Fold | 5 | 24.89 |
| LOOCV | 3,040 | 24.92 |
| Time Series Split | 5 | 24.95 |

## 🔍 Key Insights

- **Best model: SVR (RBF)**, with the lowest CV RMSE and highest test R² (0.939). On average it is off by about **16 lakh INR** per house.
- **Regularization works:** an unpruned decision tree scores train R² 1.00 vs test 0.87 (clear overfitting). After limiting depth and leaf size, the gap shrinks to 0.02.
- **Cross-validation is stable:** all four CV strategies give RMSE between 24.89 and 24.95, so performance estimates are reliable.
- **Non-linear beats linear** (0.939 vs 0.920 R²), but the gain is modest, so the linear relationship is already strong.
- **Most important drivers:** `area_sqft` and `location_score` have by far the largest effect; `property_age` and `crime_rate_index` reduce price.
- **Lasso kept all features**, which means every feature carries some signal.

## 🖼️ Visualizations

| | |
|---|---|
| ![Price distribution](plots/01_price_dist.png) | ![Correlation](plots/02_corr.png) |
| ![Alpha curves](plots/03_alpha_curves.png) | ![Coefficients](plots/04_coefficients.png) |
| ![CV comparison](plots/05_cv_compare.png) | ![Tree depth](plots/06_tree_depth.png) |
| ![RF importance](plots/07_rf_importance.png) | ![Model comparison](plots/08_model_compare.png) |

![Best model: actual vs predicted](plots/09_best_model.png)

## 💻 Streamlit App

The app loads `model.pkl` and lets you enter property details to get an **estimated price in INR**, plus a tab with the model comparison table and chart.

> 📸 Add app screenshots here, e.g. `![App](screenshots/app.png)`

## 📁 Repository Structure

```
.
├── PR2_Robust_Regression.ipynb                      # Full notebook (Parts A–H, executed)
├── app.py                                           # Streamlit app
├── model.pkl                                        # Best saved model (SVR RBF)
├── results.csv                                      # Model comparison table
├── Advanced_Regression_HousePrice_Dataset_3800.xlsx # Dataset
├── plots/                                           # All saved charts
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Usage

```bash
# 1. Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook PR2_Robust_Regression.ipynb

# 5. Run the app
streamlit run app.py
```

> `model.pkl` was trained with **scikit-learn 1.8.0**. Use the same version (pinned in `requirements.txt`) to avoid loading issues, or re-run the notebook to regenerate it.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit · Jupyter · joblib

## 🔗 Links

| Resource | Link |
|---|---|
| 🚀 Live Streamlit App | _add your URL_ |
| 🎥 Video Explanation | _add your Google Drive / YouTube (unlisted) link_ |

## 👤 Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username) · LinkedIn: _add link_

---

<div align="center">⭐ If you found this useful, consider starring the repo!</div>
