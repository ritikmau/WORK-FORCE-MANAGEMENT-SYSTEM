
````
# 🚀 Workforce Management System

This project applies multiple machine learning algorithms to **predict workforce allocation** and selects the **best-performing model** through evaluation and hyperparameter tuning.

---

## 1️⃣ Overview
This script:
- 🔄 **Preprocesses** workforce allocation data
- 🤖 **Trains** multiple machine learning models
- 📊 **Evaluates** them using **R²**, **RMSE**, and **MAE**
- 🏆 **Identifies** the best model
- ⚙️ **Performs hyperparameter tuning** for optimal performance

---

## 2️⃣ Dataset
- **File:** `allocations.csv`
- **Features:**
  - First **3 columns** → *Categorical* (encoded using `LabelEncoder`)
  - **4th column** → *Numeric*
- **Target:** Last column (*Workforce allocation value*)

---

## 3️⃣ Models Used
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor 🌲
- Gradient Boosting Regressor 🌟
- K-Nearest Neighbors (KNN) Regressor 👥
- Support Vector Regressor (SVR)

---

## 4️⃣ Evaluation Metrics
| Metric  | Description |
|---------|-------------|
| **R²**  | Measures explained variance |
| **RMSE** | Penalizes large errors |
| **MAE** | Measures average absolute errors |

---

## 5️⃣ Hyperparameter Tuning
Performed using **GridSearchCV** for:
- **KNN:** `n_neighbors`, `weights`, `p`
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`
- **Gradient Boosting:** `n_estimators`, `learning_rate`, `max_depth`
- **Ridge & Lasso:** `alpha`
- **SVR:** `C`, `epsilon`, `kernel`

---

## 6️⃣ How to Run
### Install Dependencies
```bash
pip install pandas numpy scikit-learn
````

### Run the Script

```bash
python wfm_all.py
```

---

## 7️⃣ Example Output

```
Random Forest -> R2: 0.9179, RMSE: 1.5695, MAE: 1.1793
Linear Regression -> R2: 0.8538, RMSE: 2.0944, MAE: 1.6335
Ridge -> R2: 0.8786, RMSE: 1.9083, MAE: 1.4961
Lasso -> R2: 0.8814, RMSE: 1.8869, MAE: 1.4540
Gradient Boosting -> R2: 0.9279, RMSE: 1.4710, MAE: 1.2168
KNN -> R2: 0.9327, RMSE: 1.4213, MAE: 1.2600
SVR -> R2: -0.1536, RMSE: 5.8839, MAE: 4.7126

Best Model: KNN | R2: 0.9327

Running Hyperparameter Tuning for KNN...
Best Parameters: {'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
Best Cross-Validated R2: 0.9363
Tuned R2: 0.9143
```

---

## 8️⃣ Repository Structure

```
WORK-FORCE-MANAGEMENT-SYSTEM/
│── wfm_all.py          # Main script
│── allocations.csv     # Dataset
│── README.md           # Project documentation
```

---

## 9️⃣ Future Enhancements

* 📈 Add feature importance visualization
* 🤖 Integrate deep learning models
* 🌐 Deploy as a web app using **Flask**/**Django**

```
