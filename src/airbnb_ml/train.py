import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from features import parse_bathroom, add_amenity_count, price_to_float, TopAmenities
from preprocessing import preprocessor, preprocessor_scaled
from xgboost import XGBRegressor

SEED = 45

df = pd.read_csv('data/raw/listings.csv')

df = df.dropna(subset=['price'])

df = parse_bathroom(df)
df = add_amenity_count(df)
df = price_to_float(df)

df['logprice'] = np.log(df['price'])

X_features = ['hosts_time_as_host_years', 'longitude',
                'latitude', 'accommodates', 'bedrooms', 'beds',
                'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
                'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month',
                'amenities_count', 'bathroom_count', 'bathroom_type', 'host_is_superhost', 
                'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'amenities']

X = df[X_features].copy()
y = df['price']
y_log = df['logprice']

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=SEED)

lower = y_train.quantile(0.01)
upper = y_train.quantile(0.99)
train_mask = y_train.between(lower, upper)
X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]

model = Pipeline([
    ('top_amenities', TopAmenities(n=40)),
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        random_state=SEED,
        n_jobs=1))
])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "regressor__n_estimators": [300, 500],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": [1.0, 0.7],
    "regressor__max_depth": [None, 20],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=kf,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print(f"Best CV MAE: ${-grid_search.best_score_:.2f}")

Path('models').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

joblib.dump(grid_search.best_estimator_, 'models/tuned_random_forest.joblib')

X_test.to_pickle('data/processed/X_test.pkl')
y_test.to_pickle('data/processed/y_test.pkl')