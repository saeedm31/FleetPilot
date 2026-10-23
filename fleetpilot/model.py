"""Purchase-likelihood model: gradient boosting, calibrated, with per-customer reasons.

XGBoost when installed (production image); scikit-learn's HistGradientBoostingClassifier
otherwise, behind the same interface. Reasons are SHAP values: exact TreeSHAP from XGBoost
(`pred_contribs`), or a simple one-feature-at-a-time approximation for the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_names, matrix, reason_text

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover - exercised only without xgboost
    HAS_XGB = False


@dataclass
class PurchaseModel:
    seed: int = 0
    backend: str = field(default="xgboost" if HAS_XGB else "sklearn-hgb")

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> PurchaseModel:
        xt, yt = matrix(train), train["y"].to_numpy()
        if self.backend == "xgboost":
            self.model_ = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2.0,
                eval_metric="aucpr",
                random_state=self.seed,
                n_jobs=2,
            )
            self.model_.fit(xt, yt)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model_ = HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=60,
                l2_regularization=1.0,
                random_state=self.seed,
            )
            self.model_.fit(xt, yt)
        # Platt calibration on later months: a score of 0.3 should mean ~30% of such customers buy
        self.cal_ = LogisticRegression(C=1e6, max_iter=1000)
        self.cal_.fit(_logit(self._raw(calib))[:, None], calib["y"].to_numpy())
        self.reference_ = xt.median()
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(matrix(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.cal_.predict_proba(_logit(self._raw(df))[:, None])[:, 1]

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature push on the log-odds of buying (positive = more likely)."""
        x = matrix(df)
        if self.backend == "xgboost":
            c = self.model_.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[:, :-1]
            return pd.DataFrame(c, columns=x.columns, index=df.index)
        base = _logit(self.model_.predict_proba(x)[:, 1])
        out = {}
        for col in x.columns:
            xr = x.copy()
            xr[col] = self.reference_[col]
            out[col] = base - _logit(self.model_.predict_proba(xr)[:, 1])
        return pd.DataFrame(out, index=df.index)

    def reasons(self, df: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Top k reasons per row, strongest first, in plain words."""
        contrib = self.contributions(df)
        x = matrix(df)
        out = []
        for idx in df.index:
            row = contrib.loc[idx].sort_values(key=np.abs, ascending=False)
            items = []
            for feat, val in row.head(k).items():
                if abs(val) < 1e-6:
                    continue
                items.append(
                    dict(
                        feature=feat,
                        effect=round(float(val), 3),
                        direction="raises" if val > 0 else "lowers",
                        text=reason_text(feat, float(x.loc[idx, feat])),
                    )
                )
            out.append(items)
        return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LogisticBaseline:
    """Simple, explainable baseline: logistic regression on the same features."""

    def fit(self, train: pd.DataFrame):
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        self.model_.fit(matrix(train).fillna(matrix(train).median()), train["y"])
        self.fill_ = matrix(train).median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(matrix(df).fillna(self.fill_))[:, 1]


def rule_of_thumb(df: pd.DataFrame) -> np.ndarray:
    """What many sales teams do today: call the oldest fleets first."""
    return df["avg_age_years"].to_numpy()


def evaluate(y: np.ndarray, score: np.ndarray, calibrated: bool = True) -> dict:
    y = np.asarray(y, float)
    order = np.argsort(-score)
    n = len(y)
    out = dict(
        auc=float(roc_auc_score(y, score)), pr_auc=float(average_precision_score(y, score)), base_rate=float(y.mean())
    )
    for top in (0.1, 0.2):
        k = int(round(n * top))
        found = y[order[:k]].sum() / max(y.sum(), 1)
        out[f"buyers_found_top{int(top * 100)}"] = float(found)
        out[f"lift_top{int(top * 100)}"] = float(found / top)
    if calibrated:
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    return out


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    df = pd.DataFrame(dict(y=y, p=p))
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    return g.round(4).to_dict("records")


__all__ = ["PurchaseModel", "LogisticBaseline", "rule_of_thumb", "evaluate", "calibration_table", "feature_names"]
