import numpy as np
import pytest

from autolabel.calibration import ConfidenceModel, fit_logistic, roc_auc


def test_roc_auc_basic_cases():
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == pytest.approx(1.0)
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([True, True, False, False])) == pytest.approx(0.0)
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([True, False, True, False])) == pytest.approx(0.5)
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([True, True])))


def test_roc_auc_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    scores = np.round(rng.random(200), 1)                 # con muchos empates
    labels = rng.random(200) < 0.4
    pos, neg = scores[labels], scores[~labels]
    expected = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    assert roc_auc(scores, labels) == pytest.approx(expected)


def test_logistic_fit_recovers_a_separable_signal():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.5).astype(float)
    informative = y * 2 + rng.normal(0, 0.8, 600)
    noise = rng.normal(0, 1, 600)
    model = fit_logistic(np.column_stack([informative, noise]), y, feature_names=("signal", "noise"))
    p = model.predict(np.column_stack([informative, noise]))
    assert roc_auc(p, y.astype(bool)) > 0.9
    assert abs(model.weights[0]) > 5 * abs(model.weights[1])        # el peso va a la señal útil
    assert p.min() >= 0 and p.max() <= 1


def test_model_roundtrip(tmp_path):
    x = np.array([[0.1, 1.0], [0.9, 0.0], [0.2, 1.0], [0.8, 0.0]])
    model = fit_logistic(x, np.array([0.0, 1.0, 0.0, 1.0]), feature_names=("a", "b"))
    model.save(tmp_path / "m.json")
    loaded = ConfidenceModel.load(tmp_path / "m.json")
    assert loaded == model
    assert loaded.predict(x) == pytest.approx(model.predict(x))


def test_feature_names_must_match_columns():
    with pytest.raises(ValueError):
        fit_logistic(np.zeros((4, 2)), np.array([0.0, 1.0, 0.0, 1.0]), feature_names=("solo_una",))
