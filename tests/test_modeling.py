import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from uhi import modeling

FAST_PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2, "random_state": 0}


def synthetic_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 1 + 0.02 * X[:, 0] - 0.01 * X[:, 1] + 0.001 * rng.normal(size=n)
    return pd.DataFrame(X, columns=["a", "b", "c", "d"]).assign(**{"UHI Index": y})


def test_prepare_training_table_drops_join_columns_duplicates_and_bad_rows():
    mapped = gpd.GeoDataFrame({
        "Longitude_left": [1.0, 2.0, 3.0, 4.0], "Latitude_left": [1.0, 2.0, 3.0, 4.0],
        "UHI Index": [1.0, 1.1, 1.2, 1.3], "index_right": [0, 1, 2, 3],
        "Longitude_right": [1.0, 2.0, 3.0, 4.0], "Latitude_right": [1.0, 2.0, 3.0, 4.0],
        "f": [5.0, 5.0, np.inf, 7.0], "is_bronx": [1, 1, 0, 0],
    }, geometry=gpd.points_from_xy([1, 2, 3, 4], [1, 2, 3, 4]))
    table = modeling.prepare_training_table(mapped)
    # row 1 duplicates row 0 in every feature column, row 2 holds inf
    assert list(table.columns) == ["UHI Index", "f"]
    assert table["UHI Index"].tolist() == [1.0, 1.3]


def test_numeric_features_excludes_target():
    X = modeling.numeric_features(pd.DataFrame({"UHI Index": [1.0], "a": [np.inf], "b": [2]}))
    assert list(X.columns) == ["a", "b"] and X.loc[0, "a"] == 0


def test_train_evaluate_predict_and_importance():
    table = synthetic_table()
    trained = modeling.train_and_evaluate(table, features=["a", "b", "c", "d"], params=FAST_PARAMS)
    assert trained.test_r2 > 0.8 and len(trained.cv_scores) == 5
    assert len(trained.test_predictions) == len(trained.y_test) == 80

    assert modeling.predict(trained, table).shape == (400,)
    importance = modeling.feature_importance(trained)
    assert importance.index[0] == "a"  # the strongest signal
    assert set(importance.index) <= {"a", "b", "c", "d"}


def test_run_rfecv_selects_the_informative_features():
    # ImportError, not only ModuleNotFoundError: an OS policy can block compiled sklearn modules
    pytest.importorskip("sklearn.feature_selection", exc_type=ImportError)
    table = synthetic_table(n=300)
    result = modeling.run_rfecv(table[["a", "b", "c", "d"]], table["UHI Index"], params=FAST_PARAMS)
    assert {"a", "b"} <= set(result["selected_features"])
    assert result["n_features"] == len(result["selected_features"])
