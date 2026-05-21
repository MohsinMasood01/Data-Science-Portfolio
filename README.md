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

- 1,002 trading days (April 2022 - April 2026), 921 sequences generated
- Total trainable parameters: 30,651
- Training loss reduced from 0.0408 (Epoch 1) to 0.0027 (Epoch 20)
- MAE: $6.46 | RMSE: $8.35 (~3% average error on prices ranging $130-$280)

**How to Run:**

1. Clone the repository
2. Install dependencies: 'pip install pandas numpy matplotlib scikit-learn tensorflow yfinance'
3. Run the notebook: 'stock_price_prediction.ipynb'
4. Data is fetched automatically, no manual download required

---

## Project 3: Customer Segmentation using K-Means Clustering
A machine learning model trained on the Kaggle Mall Customer dataset to segment customers into distinct groups based on their annual income and spending behaviour using unsupervised learning.

**Tools & Libraries:**

- Python
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib & seaborn (data visualization)
- scikit-learn (KMeans clustering, StandardScaler)

**Features:**

- Explores feature distributions using histograms and scatter plots
- Scales features using StandardScaler to prevent bias from differing value ranges
- Determines optimal number of clusters using the Elbow Method (inertia vs K)
- K-Means clustering with K=5 confirmed by both Elbow Method and visual inspection
- Visualizes 5 customer segments with color-coded scatter plot and centroids
- Describes each segment using average age, income, and spending score

**Notes:**

- 200 customer samples, 2 features used for clustering (Annual Income, Spending Score)
- 5 segments identified: Average, High Income High Spending, Low Income High Spending, High Income Low Spending, Low Income Low Spending
- Youngest group (avg age 25) spent the most relative to income, identified as impulsive buyers
- Oldest groups (avg age 41-45) were the most financially conservative

**How to Run:**

1. Clone the repository
2. Install dependencies: 'pip install pandas numpy matplotlib seaborn scikit-learn'
3. Download the Mall Customer dataset from Kaggle and place *'Mall_Customers.csv'* in the same directory
4. Run the notebook: 'customer_segmentation.ipynb'

---

## Project 4: Movie Rating Prediction
A collaborative filtering model trained on the MovieLens 100K dataset to predict how a user might rate a movie they have not seen yet, using SVD matrix factorization to discover hidden patterns in user preferences.

**Tools & Libraries:**

- Python
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib & seaborn (data visualization)
- scikit-learn (evaluation metrics)
- scipy (SVD matrix factorization)

**Features:**

- Merges ratings and movie title data for readability
- Explores rating distribution and most rated movies via EDA
- Builds a 610 x 9,719 user-movie utility matrix
- Applies SVD with k=50 latent factors to discover hidden user preference patterns
- Reconstructs full predicted ratings matrix for all user-movie combinations
- Evaluates using MAE and RMSE on known ratings
- Recommendation function filters already-rated movies and returns top predicted titles per user

**Notes:**

- 100,836 ratings across 610 users and 9,742 movies
- 50 latent factors learned purely from rating behaviour, no genre or metadata used
- MAE: 1.53 | RMSE: 2.00 on a 5-star scale
- Recommendations for User 1 included: Die Hard (1988), The Godfather Part II (1974), and Jaws (1975); consistent with their known preference for crime and thriller films

**How to Run:**

1. Clone the repository
2. Install dependencies: 'pip install pandas numpy matplotlib seaborn scikit-learn scipy'
3. Download the MovieLens 100K dataset from Kaggle and place *'ratings.csv'* and *'movies.csv'* in the same directory
4. Run the notebook: 'movie_rating_prediction.ipynb'
