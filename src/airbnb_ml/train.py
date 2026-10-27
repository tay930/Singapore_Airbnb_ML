import pandas as pd
import json
import joblib
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from features import parse_bathroom, add_amenity_count, add_amenity_features, price_to_float
from preprocessing import preprocessor, numerical_columns, categorical_columns, numerical_pipeline, categorical_pipeline

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

amenity_counts = Counter(
    amenity for value in X_train["amenities"].dropna()
    if isinstance(value, str)
    for amenity in json.loads(value)
)

top_40_names = [amenity for amenity, count in amenity_counts.most_common(40)]

train_amenity_lists = X_train["amenities"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else []
)

test_amenity_lists = X_test["amenities"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else []
)

for amenity in top_40_names:
    column_name = f"has_{amenity}"

    X_train[column_name] = train_amenity_lists.apply(
        lambda items: int(amenity in items)
    )

    X_test[column_name] = test_amenity_lists.apply(
        lambda items: int(amenity in items)
    )

X_train = X_train.drop(columns=['amenities'])
X_test = X_test.drop(columns=['amenities'])

amenity_columns = [
    col for col in X_train.columns
    if col.startswith("has_")
]

base_numerical_columns = [
    col for col in numerical_columns
    if not col.startswith("has_")
]

numerical_columns_final = (
    base_numerical_columns + amenity_columns
)

lower = y_train.quantile(0.01)
upper = y_train.quantile(0.99)
train_mask = y_train.between(lower, upper)
X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns_final),
    ("categorical", categorical_pipeline, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

Path('models').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

joblib.dump(model, 'models/random_forest_100_trees.joblib')

X_test.to_pickle('data/processed/X_test.pkl')
y_test.to_pickle('data/processed/y_test.pkl')