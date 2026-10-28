import numpy as np
import pandas as pd
import json
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

def parse_bathroom(df):
    df = df.copy()
    df["bathroom_type"] = np.where(
        df["bathrooms_text"].str.contains("private", case=False, na=False),
        "private",
        "shared"
    )

    df["bathroom_count"] = np.where(
        df['bathrooms_text'].str.contains("half-bath", case=False, na=False),
        0.5,
        pd.to_numeric(
            df['bathrooms_text'].str.extract(r"(\d+(?:\.\d+)?)", expand=False),
            errors="coerce"
        )
    )

    return df

def add_amenity_count(df):
    df = df.copy()

    df["amenities_count"] = df["amenities"].apply(
        lambda x: len(json.loads(x)) if isinstance(x, str) else np.nan
    )

    return df

def add_amenity_features(df, top_40_names):
    df = df.copy()

    amenity_lists = df["amenities"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    for amenity in top_40_names:
        df[f"has_{amenity}"] = amenity_lists.apply(
            lambda items: int(amenity in items)
        )

    return df

def price_to_float(df):
    df = df.copy()

    df['price'] = pd.to_numeric(df["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df['price'] = df['price'].astype(float)

    return df


class TopAmenities(BaseEstimator, TransformerMixin):
    def __init__(self, n=40):
        self.n = n

    def fit(self, X, y=None):
        amenities = X["amenities"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

        counts = Counter(
            amenity
            for items in amenities
            for amenity in items
        )

        self.top_amenities_ = [
            amenity for amenity, count in counts.most_common(self.n)
        ]

        return self

    def transform(self, X):
        X = X.copy()

        amenities = X["amenities"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

        for amenity in self.top_amenities_:
            X[f"has_{amenity}"] = amenities.apply(
                lambda items: int(amenity in items)
            )
            
        X = X.drop(columns=["amenities"])

        return X

