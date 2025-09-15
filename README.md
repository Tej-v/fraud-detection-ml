# Fraudulent Transaction Detection Using Machine Learning

An end-to-end machine learning project that detects fraudulent financial transactions using ensemble methods on highly imbalanced data. This project demonstrates advanced techniques for handling class imbalance, feature engineering, and model calibration.

## Project Overview

This project analyzes financial transaction data to identify fraudulent activities using multiple machine learning algorithms. With fraudulent transactions representing only **0.13%** of the dataset, the project emphasizes techniques for handling severe class imbalance while maintaining high precision and recall.

## Key Highlights

- Handled 99.87% class imbalance using SMOTE-Tomek resampling and class weighting
- Achieved ROC-AUC of 0.987 across ensemble models
- Implemented 4 ML algorithms: Logistic Regression, Random Forest, XGBoost, and LightGBM
- Applied probability calibration using isotonic regression for reliable fraud probability scores
- Engineered domain-specific features to capture balance discrepancies and temporal fraud patterns

## Tech Stack

- **Python 3.x**
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Imbalance Handling**: Imbalanced-learn (SMOTE-Tomek)

## Dataset

The dataset contains **6.3 million** financial transactions with the following features:
- Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- Transaction amount
- Account balances (origin and destination, before and after transaction)
- Temporal information (step-based timestamps)
- Fraud labels (binary classification)

## Feature Engineering

Created multiple engineered features to improve fraud detection:

### 1. Balance Discrepancy Features
- `origin_diff`: Detects inconsistencies in sender's balance changes
- `dest_diff`: Identifies anomalies in receiver's balance updates

### 2. Transaction Flags
- `is_large_amount`: Flags transactions in top 1% by amount
- `is_origin_zero`: Identifies payments from zero-balance accounts
- `is_merchant_dest`: Distinguishes merchant vs. customer transactions

### 3. Temporal Features
- `day_number`: Day of month (1-31)
- `hour_of_day`: Hour of day (0-23)

### 4. Categorical Encoding
- One-hot encoding for transaction types

## Model Performance

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC |
|-------|------------------|----------------|----------|---------|
| Logistic Regression | 0.86 | 0.41 | 0.56 | 0.987 |
| Random Forest | High | Moderate | Strong | 0.98+ |
| XGBoost | High | Moderate | Strong | 0.98+ |
| LightGBM | High | Moderate | Strong | 0.98+ |

**Best Model**: XGBoost with threshold tuning (threshold = 0.3)
- Improved recall for detecting more fraudulent transactions
- Maintained high precision to minimize false alarms

## Key Findings

### Top Fraud Predictors

1. **Transaction Type**: 100% of frauds occur in TRANSFER and CASH_OUT transactions
2. **Transaction Amount**: Fraudulent transactions involve significantly larger amounts
3. **Balance Mismatches**: Fraudulent transactions show abnormal balance behavior (origin_diff ≈ 0)
4. **Temporal Patterns**: Fraud spikes during early morning hours (0-6 AM) and end of month
5. **Zero-Balance Origins**: Payments from zero-balance accounts are highly suspicious

## Methodology

### 1. Data Cleaning
- Handled missing values in merchant account balances (34% missing by design)
- Created `is_merchant_dest` flag instead of imputing missing balances
- Retained outliers as they often represent fraudulent activity
- Addressed multicollinearity through feature engineering

### 2. Exploratory Data Analysis
- Analyzed fraud distribution across transaction types
- Examined amount distributions for fraud vs. non-fraud
- Investigated temporal patterns in fraudulent activities
- Visualized balance behavior discrepancies

### 3. Model Training Strategy
- **Split**: 60% training, 20% calibration, 20% validation
- **Imbalance Handling**: Class weighting and scale_pos_weight for tree-based models
- **Calibration**: Isotonic regression for probability calibration
- **Threshold Tuning**: Adjusted decision threshold from 0.5 to 0.3 for better recall

### 4. Evaluation Metrics
- **ROC-AUC**: Primary metric for model discrimination ability
- **Precision-Recall**: Critical for imbalanced classification
- **F1-Score**: Balance between precision and recall
- Avoided accuracy due to severe class imbalance

## Business Recommendations

### Preventive Measures
1. **Real-time Monitoring**: Deploy models for live transaction scoring
2. **Enhanced Authentication**: Multi-factor verification for high-risk transactions
3. **Transaction Limits**: Stricter controls for new/zero-balance accounts
4. **Anomaly Detection**: Flag unusual patterns (time, amount, balance mismatches)
5. **Metadata Logging**: Capture device, IP, and location data for forensics

### Success Metrics
- Monitor fraud loss reduction over time
- Track model recall (fraud detection rate) and precision (false alarm rate)
- Compare fraud attempts vs. successful frauds pre/post-implementation
- Analyze customer dispute trends

## Getting Started

### Installation

Clone repository
git clone https://github.com/Tej-v/fraud-detection-ml.git
cd fraud-detection-ml

Install dependencies
pip install -r requirements.txt

### Dependencies


pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
imbalanced-learn==0.11.0


### Usage

Run the fraud detection model
python fraud_detection.py


**Note**: Update the dataset path in the code:
data = pd.read_csv("path/to/your/Fraud.csv")


## Project Structure

fraud-detection-ml/
├── fraud_detection.py # Main project code
├── README.md # Project documentation


## Skills Demonstrated

- Data cleaning and preprocessing
- Handling highly imbalanced datasets
- Advanced feature engineering
- Multiple ML algorithm implementation
- Model calibration and threshold tuning
- Business-focused evaluation metrics
- Statistical analysis and visualization
- Domain knowledge application in financial fraud

## Contact

**Teja Velpu**
- GitHub: [@Tej-v](https://github.com/Tej-v)
- LinkedIn: [www.linkedin.com/in/tejaswini-velpula]

## License

This project is open source and available under the [MIT License](LICENSE).
