import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numerical_columns = ['hosts_time_as_host_years', 'longitude',
                    'latitude', 'accommodates', 'bedrooms', 'beds',
                    'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
                    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                    'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month',
                    'amenities_count', 'bathroom_count', 'has_Wifi', 'has_Air conditioning', 'has_Essentials', 
                    'has_Hot water', 'has_TV', 'has_Hair dryer', 'has_Iron', 'has_Hangers', 'has_Kitchen', 
                    'has_Bed linens', 'has_Smoke alarm', 'has_Dedicated workspace', 'has_Washer', 'has_Elevator', 
                    'has_Self check-in', 'has_Long term stays allowed', 'has_Luggage dropoff allowed', 'has_Shampoo', 
                    'has_Microwave', 'has_Refrigerator', 'has_Dishes and silverware', 'has_Fire extinguisher', 
                    'has_Hot water kettle', 'has_Dryer', 'has_Building staff', 'has_Shower gel', 'has_Stove', 'has_Pool', 
                    'has_Cooking basics', 'has_Clothing storage', 'has_Heating', 'has_Free parking on premises', 
                    'has_First aid kit', 'has_Exterior security cameras on property', 'has_Bidet', 
                    'has_Extra pillows and blankets', 'has_Dining table', 'has_Room-darkening shades', 
                    'has_Lock on bedroom door', 'has_Gym']

categorical_columns = ['bathroom_type', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

