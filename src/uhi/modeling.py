"""Training-table preparation, feature selection, XGBoost training / evaluation and prediction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

# Columns created by the nearest-pixel join that are not features
JOIN_COLUMNS = ["Longitude_left", "Latitude_left", "index_right", "Longitude_right", "Latitude_right"]


def prepare_training_table(mapped, target=config.TARGET):
    """
    Turn the mapped training points into a modelling table.

    Drops the geometry, duplicate feature rows, rows with NaN / inf, the join columns and the
    borough flag, and converts any object column to numbers.
    """
    table = pd.DataFrame(mapped.drop(columns="geometry"))
    feature_columns = [c for c in table.columns if c not in JOIN_COLUMNS + [target]]
    table = table.drop_duplicates(subset=feature_columns, keep="first")
    table = table.dropna()
    table = table[~table.isin([np.inf, -np.inf]).any(axis=1)]
    table = table.drop(columns=JOIN_COLUMNS + ["is_bronx"])

    for col in table.select_dtypes(include=["object"]).columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
        table[col] = table[col].fillna(table[col].mean()).astype(int)
    return table


def numeric_features(table, target=config.TARGET):
    """All numeric columns except the target, with inf replaced by NaN and NaN by 0."""
    X = table.select_dtypes(include=["number"]).drop(columns=[target])
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def run_rfecv(X, y, params=None, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """
    Recursive feature elimination with 5-fold CV (R² scoring, one feature removed per step).

    Returns a dict with the selected feature names and the hold-out R² of a model refit on them.
    params=None uses default XGBoost parameters; pass config.XGB_PARAMS for the tuned model.
    """
    # Imported here so that training and prediction do not depend on sklearn.feature_selection
    from sklearn.feature_selection import RFECV

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = XGBRegressor(**params) if params else XGBRegressor(objective="reg:squarederror", random_state=seed)
    selector = RFECV(estimator=model, cv=5, scoring="r2", step=1).fit(X_train, y_train)
    model.fit(selector.transform(X_train), y_train)
    return {
        "n_features": int(selector.n_features_),
        "selected_features": list(X_train.columns[selector.support_]),
        "test_r2": model.score(selector.transform(X_test), y_test),
        "selector": selector,
    }


@dataclass
class TrainedModel:
    model: XGBRegressor
    scaler: StandardScaler
    features: list
    cv_scores: np.ndarray
    train_r2: float
    test_r2: float
    y_test: np.ndarray
    test_predictions: np.ndarray


def train_and_evaluate(table, features=config.SELECTED_FEATURES, params=config.XGB_PARAMS,
                       target=config.TARGET, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """
    Split 80/20, standardise, evaluate XGBoost with 5-fold CV on the training split, then fit on
    the whole training split and score the training and hold-out sets.
    """
    X = table[features].values
    y = table[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                        shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, scoring="r2",
                                cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    return TrainedModel(model=model, scaler=scaler, features=list(features), cv_scores=cv_scores,
                        train_r2=r2_score(y_train, model.predict(X_train)),
                        test_r2=r2_score(y_test, test_predictions),
                        y_test=y_test, test_predictions=test_predictions)


def predict(trained, table):
    """Predictions for a table that contains the trained features."""
    return trained.model.predict(trained.scaler.transform(table[trained.features].values))


def feature_importance(trained, importance_type="weight"):
    """Importance of every feature (XGBoost names them f0, f1, ...), sorted from high to low."""
    scores = trained.model.get_booster().get_score(importance_type=importance_type)
    named = {trained.features[int(key[1:])]: value for key, value in scores.items()}
    return pd.Series(named, name=importance_type).sort_values(ascending=False)
