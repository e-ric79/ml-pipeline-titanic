# Complete ML Pipeline: Titanic Survival Prediction

## Overview
A production-ready machine learning pipeline demonstrating industry best practices for binary classification. This project showcases the complete workflow from data exploration to model deployment, achieving 83.84% accuracy in predicting Titanic passenger survival.

## Project Highlights

- ✅ **Professional ML Workflow** - End-to-end pipeline following best practices
- ✅ **Model Comparison** - Systematic evaluation of multiple algorithms
- ✅ **Cross-Validation** - Robust performance estimation using 5-fold CV
- ✅ **Hyperparameter Tuning** - Grid search optimization across 108 configurations
- ✅ **Model Persistence** - Saved model ready for deployment
- ✅ **Reproducible Results** - Fixed random seeds for consistency

## Final Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | **83.84%** |
| **Model Type** | Random Forest (Tuned) |
| **Cross-Validation** | 5-Fold |
| **Training Samples** | 891 passengers |
| **Features Used** | 6 |

## Pipeline Architecture

### 8-Step Professional Workflow

#### 1. Data Loading & Exploration
- Loaded Titanic dataset (891 passengers)
- Analyzed target distribution (38.38% survival rate)
- Identified missing values and data types
- Visualized survival patterns by class

#### 2. Data Cleaning
- Selected 6 relevant features (Pclass, Sex, Age, SibSp, Parch, Fare)
- Filled missing Age values with median
- Filled missing Fare values with median
- Encoded Sex as numeric (male=1, female=0)
- Result: Zero missing values

#### 3. Feature Analysis
- Calculated correlations with survival outcome
- **Sex:** Strongest predictor (correlation: -0.54)
- **Fare:** Positive correlation (wealth matters)
- **Pclass:** Negative correlation (class hierarchy)
- Created correlation heatmap for visualization

#### 4. Model Selection & Comparison
Tested three algorithms with default parameters:

| Model | Initial Accuracy |
|-------|-----------------|
| Logistic Regression | 79.00% |
| Decision Tree | 80.45% |
| Random Forest | 81.00% |

**Winner:** Random Forest selected for further optimization

#### 5. Cross-Validation
Performed 5-fold cross-validation for robust evaluation:

| Model | Mean CV Score | Std Dev |
|-------|---------------|---------|
| Logistic Regression | 79.00% | ±1.2% |
| Decision Tree | 80.45% | ±2.1% |
| **Random Forest** | **81.00%** | **±1.5%** |

**Key Insight:** Random Forest shows best accuracy AND stability

#### 6. Hyperparameter Tuning
Grid Search with 5-fold CV testing 108 combinations:

**Parameters Tested:**
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

**Optimal Configuration Found:**
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 1
}
```

**Performance Improvement:**
- Default Random Forest: 81.00%
- Tuned Random Forest: 83.84%
- **Improvement: +2.84%**

#### 7. Final Model Evaluation

**Feature Importance Rankings:**
1. **Sex (0.33)** - Gender was decisive factor
2. **Fare (0.25)** - Wealth/cabin location mattered
3. **Age (0.22)** - Age influenced survival
4. **Pclass (0.10)** - Passenger class effect
5. **SibSp (0.06)** - Family relationships
6. **Parch (0.04)** - Parent/child aboard

**Model Performance:**
- 5-Fold CV Mean: 83.84%
- Standard Deviation: ±1.5%
- Consistent across all folds

#### 8. Model Persistence

**Saved Artifacts:**
- `titanic_survival_model.pkl` - Trained Random Forest model
- `feature_names.pkl` - Feature column names

**Deployment Test Successful:**
- Model loads correctly
- Makes predictions on new data
- Returns probability estimates

## Model Evolution Journey
```
Step 1: Baseline (Logistic Regression)     → 79.00%
Step 2: Tree-Based (Decision Tree)         → 80.45%
Step 3: Ensemble (Random Forest Default)   → 81.00%
Step 4: Optimized (Random Forest Tuned)    → 83.84% ✓
```

**Total improvement from baseline: +4.84%**

## Key Insights

### What Drives Survival?

1. **Gender is Critical**
   - Females had ~74% survival rate
   - Males had ~19% survival rate
   - "Women and children first" policy evident

2. **Class Matters**
   - 1st class: ~63% survival
   - 2nd class: ~47% survival
   - 3rd class: ~24% survival
   - Better cabin locations = closer to lifeboats

3. **Wealth Correlates with Survival**
   - Higher fare = better survival chances
   - Reflects both class and cabin quality

4. **Age Factor**
   - Children prioritized
   - Elderly had lower survival rates

### Why This Model Works

✓ **Feature Selection:** Focused on most predictive features
✓ **Ensemble Method:** Random Forest combines multiple decision trees
✓ **Hyperparameter Tuning:** Optimized for best performance
✓ **Cross-Validation:** Robust evaluation prevents overfitting

## Technical Stack

### Languages & Libraries
```python
- Python 3.x
- pandas (data manipulation)
- NumPy (numerical computing)
- scikit-learn (machine learning)
- matplotlib & seaborn (visualization)
- joblib (model persistence)
```

### Machine Learning Techniques
- **Classification:** Binary prediction (survived/died)
- **Ensemble Learning:** Random Forest (100 trees)
- **Cross-Validation:** 5-fold stratified
- **Hyperparameter Optimization:** Grid Search
- **Model Evaluation:** Accuracy, precision, recall, F1-score

## Repository Structure
```
├── ml_pipeline_titanic_complete.ipynb    # Complete pipeline notebook
├── titanic_survival_model.pkl            # Saved trained model
├── feature_names.pkl                     # Feature metadata
├── Titani-Dataset.csv                    # Training dataset
└── README.md                             # This file
```

## How to Use This Model

### 1. Load the Model
```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load('titanic_survival_model.pkl')
features = joblib.load('feature_names.pkl')
```

### 2. Prepare New Data
```python
# Example passenger
new_passenger = pd.DataFrame({
    'Pclass': [1],          # 1st class
    'Sex': [0],             # Female
    'Age': [25],            # 25 years old
    'SibSp': [0],           # No siblings/spouse
    'Parch': [0],           # No parents/children
    'Fare': [100]           # $100 ticket
})
```

### 3. Make Prediction
```python
# Predict survival
prediction = model.predict(new_passenger)[0]
probability = model.predict_proba(new_passenger)[0]

print(f"Prediction: {'Survived' if prediction == 1 else 'Died'}")
print(f"Confidence: {probability[1]:.1%}")
```

## Best Practices Demonstrated

### Data Science Best Practices
✓ Exploratory data analysis before modeling
✓ Proper train/test methodology
✓ Cross-validation for robust estimates
✓ Multiple model comparison
✓ Systematic hyperparameter tuning
✓ Feature importance analysis
✓ Model documentation and persistence

### Code Best Practices
✓ Modular step-by-step approach
✓ Reproducible results (random seeds)
✓ Visualization of key insights
✓ Error-free execution
✓ Production-ready output

## Limitations & Future Work

### Current Limitations
- Binary classification only (survived/died, no survival time)
- Limited to 6 features (excluded Name, Ticket, Cabin complexity)
- Historical data only (1912 Titanic disaster)
- No feature engineering applied (kept simple for demonstration)

### Potential Improvements
1. **Feature Engineering**
   - Extract titles from names (Mr., Mrs., Master)
   - Create family size feature (SibSp + Parch)
   - Cabin deck extraction
   - Age binning (child, adult, senior)

2. **Advanced Models**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks
   - Ensemble stacking

3. **Deployment**
   - Create REST API (Flask/FastAPI)
   - Build web interface
   - Docker containerization
   - Cloud deployment (AWS/GCP/Azure)

4. **Monitoring**
   - Track prediction accuracy over time
   - Detect data drift
   - A/B testing of model versions


## Comparison with Other Approaches

| Approach | Accuracy | Pros | Cons |
|----------|----------|------|------|
| Simple Logistic Regression | 79% | Fast, interpretable | Limited accuracy |
| Single Decision Tree | 80% | Visual, intuitive | Prone to overfitting |
| Default Random Forest | 81% | Good baseline | Unoptimized |
| **This Pipeline (Tuned RF)** | **84%** | **Best accuracy, production-ready** | **More complex** |

## Conclusion

This project demonstrates a complete, professional machine learning pipeline achieving **83.84% accuracy** in predicting Titanic passenger survival. The systematic approach—from data exploration through model deployment—showcases industry best practices and serves as a template for future classification projects.

**Key Achievement:** Improved baseline accuracy by 4.84% through systematic model selection, cross-validation, and hyperparameter tuning.

---

## About This Project

**Context:** Part of my machine learning learning journey as a Computer Science student

**Learning Approach:** Self-directed study with AI-assisted learning (Claude AI for guidance, debugging, and best practice recommendations)

**Skills Demonstrated:**
- Data preprocessing and cleaning
- Exploratory data analysis
- Machine learning model development
- Cross-validation and hyperparameter tuning
- Model evaluation and comparison
- Model persistence and deployment preparation
- Technical documentation

---

*For questions or collaboration opportunities, feel free to reach out via GitHub.*
