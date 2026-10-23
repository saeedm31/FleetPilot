"""Customer segments: k-means on what describes a fleet, named in plain words."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENT_FEATURES = [
    "log_fleet",
    "avg_age_years",
    "km_per_truck_year",
    "our_share",
    "fleet_growth_12m",
    "service_cost_per_truck",
]

_WORDS = {
    "log_fleet": ("large fleets", "small fleets"),
    "avg_age_years": ("ageing trucks", "young trucks"),
    "km_per_truck_year": ("high mileage", "low mileage"),
    "our_share": ("loyal to us", "mostly competitor trucks"),
    "fleet_growth_12m": ("growing", "shrinking"),
    "service_cost_per_truck": ("high service costs", "low service costs"),
}


class Segmenter:
    def __init__(self, k: int = 5, seed: int = 0):
        self.k, self.seed = k, seed

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(
            dict(
                log_fleet=np.log(df["fleet_size"].clip(lower=1)),
                avg_age_years=df["avg_age_years"],
                km_per_truck_year=df["km_per_truck_year"].fillna(
                    self.km_fill_ if hasattr(self, "km_fill_") else df["km_per_truck_year"].median()
                ),
                our_share=df["our_share"],
                fleet_growth_12m=df["fleet_growth_12m"].clip(-0.5, 1.0),
                service_cost_per_truck=df["service_cost_per_truck"],
            )
        )
        return x[SEGMENT_FEATURES]

    def fit(self, df: pd.DataFrame) -> Segmenter:
        self.km_fill_ = float(df["km_per_truck_year"].median())
        x = self._x(df)
        self.scaler_ = StandardScaler().fit(x)
        z = self.scaler_.transform(x)
        self.km_ = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(z)
        self.names_ = self._name(df, self.km_.predict(z))
        return self

    def predict(self, df: pd.DataFrame) -> list[str]:
        z = self.scaler_.transform(self._x(df))
        return [self.names_[c] for c in self.km_.predict(z)]

    def _name(self, df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
        names, used = {}, set()
        centers = self.km_.cluster_centers_
        for c in range(self.k):
            order = np.argsort(-np.abs(centers[c]))
            words = [_WORDS[SEGMENT_FEATURES[j]][0 if centers[c][j] > 0 else 1] for j in order[:2]]
            name = (words[0] + ", " + words[1]).capitalize()
            if name in used:
                top_ind = df.loc[labels == c, "industry"].mode().iat[0]
                name = f"{name} ({top_ind})"
            used.add(name)
            names[c] = name
        return names

    def profile(self, df: pd.DataFrame, segments: list[str]) -> pd.DataFrame:
        d = df.assign(segment=segments)
        g = d.groupby("segment").agg(
            customers=("cid", "size"),
            trucks=("fleet_size", "sum"),
            avg_fleet=("fleet_size", "mean"),
            avg_age_years=("avg_age_years", "mean"),
            our_share=("our_share", "mean"),
            main_industry=("industry", lambda s: s.mode().iat[0]),
        )
        return g.reset_index().round(2)
