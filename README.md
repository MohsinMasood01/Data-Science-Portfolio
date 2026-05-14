# Data Science Portfolio
A collection of Data Science and Machine Learning projects completed as part of the Arch Technologies Data Science Internship. Each project follows the full data science pipeline: data collection, preprocessing, exploratory data analysis, model building, and evaluation.

---

## Project 1: Titanic Survival Classification
A machine learning model trained on the Kaggle Titanic dataset to predict whether a passenger survived or not based on features like age, gender, ticket class, and fare.

**Tools & Libraries:**

- Python
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib & seaborn (data visualization)
- scikit-learn (model building and evaluation)

**Features:**

- Handles missing values using grouped median imputation and mode filling
- Label encodes binary categorical variables (Sex) and One-Hot encodes multi-class variables (Embarked)
- Drops irrelevant columns (Name, Ticket, Cabin, PassengerId)
- 80/20 train/validation split for model evaluation
- Random Forest classifier aggregating predictions from 100 decision trees
- Evaluated using accuracy score and confusion matrix
- Submission generated for Kaggle leaderboard evaluation

**Notes:**

- 891 training samples, 418 test samples
- Gender and passenger class were the strongest predictors of survival
- Validation accuracy: 83.24% | Kaggle leaderboard score: 75.12%

**How to Run:**

1. Clone the repository
2. Install dependencies: 'pip install pandas numpy matplotlib seaborn scikit-learn'
3. Download the *'Titanic dataset'* from Kaggle and place 'train.csv' and 'test.csv' in the same directory
4. Run the notebook: 'titanic_classification.ipynb'

---

## Project 2: Stock Price Prediction
A deep learning model trained on 4 years of Apple Inc. (AAPL) historical stock data fetched directly from Yahoo Finance to predict future closing prices.

**Tools & Libraries:**

- Python
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (data visualization)
- scikit-learn (preprocessing and evaluation)
- TensorFlow/Keras (LSTM model)
- yfinance (stock data fetching)

**Features:**

- Fetches 4 years of real-time AAPL stock data directly from Yahoo Finance
- Engineers lag features (Previous Close, Price Change) and moving averages (MA7, MA21)
- Normalizes closing prices to 0-1 range using MinMaxScaler
- Generates 60-day sliding window sequences for LSTM input
- Chronological 80/20 train/validation split to preserve temporal order
- Two-layer LSTM neural network with Dropout regularization
- Reverses normalization on predictions for real USD price evaluation

**Notes:**

- 1,002 trading days (April 2022 — April 2026), 921 sequences generated
- Total trainable parameters: 30,651
- Training loss reduced from 0.0408 (Epoch 1) to 0.0027 (Epoch 20)
- MAE: $6.46 | RMSE: $8.35 (~3% average error on prices ranging $130—$280)

**How to Run:**

1. Clone the repository
2. Install dependencies: 'pip install pandas numpy matplotlib scikit-learn tensorflow yfinance'
3. Run the notebook: 'stock_price_prediction.ipynb'
4. Data is fetched automatically, no manual download required
