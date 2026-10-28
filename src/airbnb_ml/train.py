import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from features import parse_bathroom, add_amenity_count, price_to_float, TopAmenities
from preprocessing import preprocessor

SEED = 45

df = pd.read_csv('data/raw/listings.csv')

df = df.dropna(subset=['price'])

df = parse_bathroom(df)
df = add_amenity_count(df)
df = price_to_float(df)

X_features = ['hosts_time_as_host_years', 'longitude',
                'latitude', 'accommodates', 'bedrooms', 'beds',
                'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
                'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month',
                'amenities_count', 'bathroom_count', 'bathroom_type', 'host_is_superhost', 
                'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'amenities']

X = df[X_features].copy()
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

lower = y_train.quantile(0.01)
upper = y_train.quantile(0.99)
train_mask = y_train.between(lower, upper)
X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]

model = Pipeline([
    ('top_amenities', TopAmenities(n=40)),
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()
    )
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = -1 * cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print(f"Scores per fold: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

model.fit(X_train, y_train)

Path('models').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

joblib.dump(model, 'models/linear_regression.joblib')

X_test.to_pickle('data/processed/X_test.pkl')
y_test.to_pickle('data/processed/y_test.pkl')