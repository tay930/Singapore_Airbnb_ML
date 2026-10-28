import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

numerical_columns = ['hosts_time_as_host_years', 'longitude',
                    'latitude', 'accommodates', 'bedrooms', 'beds',
                    'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
                    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                    'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month',
                    'amenities_count', 'bathroom_count']

categorical_columns = ['bathroom_type', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, make_column_selector(dtype_include=np.number)),
    ("categorical", categorical_pipeline, make_column_selector(dtype_include=object))
])

