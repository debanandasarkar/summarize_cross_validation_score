from summarize_cross_validation_score import __version__
from summarize_cross_validation_score import summarize_cross_validation_score
import numpy as np
import pandas as pd

def test_version():
    assert __version__ == '0.1.0'

def test_summarize_cv_scores():
    toy_score = {
        "fit_time": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "score_time": np.array([1, 2, 3, 4, 5]),
        "test_accuracy": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "train_accuracy": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "test_f1": np.array([0.1, 0.1, 0.2, 0.1, 0.1]),
        "train_f1": np.array([0.1, 0.3, 0.1, 0.1, 0.1]),
    }

    expected = {
        "classifier_name": ["toy_test"],
        "mean_fit_time": [0.3],
        "mean_score_time": [3],
        "mean_validation_accuracy": [0.5],
        "mean_train_accuracy": [0.5],
        "mean_validation_f1": [0.12],
        "mean_train_f1": [0.14],
        "std_fit_time": [0.158114],
        "std_score_time": [1.581139],
        "std_validation_accuracy": [0.0],
        "std_train_accuracy": [0.0],
        "std_validation_f1": [0.044721],
        "std_train_f1": [0.089443],
    }

    assert isinstance(
        summarize_cross_validation_score.summarize_cv_scores(toy_score, "toy_test"), pd.DataFrame
    ), "Check data structure"
    assert (
        int(
            (
                np.round(summarize_cross_validation_score.summarize_cv_scores(toy_score, "toy_test"), 4)
                == np.round(pd.DataFrame(data=expected), 4)
            ).T.sum()
        )
        == 13
    ), "Check function logic"