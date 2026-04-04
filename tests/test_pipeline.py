import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_pipeline, evaluate

SAMPLE_REVIEWS = [
    "This movie was absolutely fantastic! Great acting and plot.",
    "Terrible film. I wasted two hours of my life. Awful.",
    "Amazing cinematography and a gripping storyline throughout.",
    "Boring, slow, and completely predictable. Very disappointing.",
    "One of the best films I have ever seen. Highly recommend.",
    "Worst movie of the year. Do not watch this garbage.",
    "Incredible performances from the entire cast. Loved it.",
    "Dull and lifeless. The script made no sense whatsoever.",
]
SAMPLE_LABELS = [1, 0, 1, 0, 1, 0, 1, 0]

class TestBuildPipeline:
    def test_pipeline_has_two_steps(self):
        p = build_pipeline(max_features=5000, C=1.0, max_iter=100)
        assert len(p.steps) == 2

    def test_pipeline_step_names(self):
        p = build_pipeline(max_features=5000, C=1.0, max_iter=100)
        names = [n for n, _ in p.steps]
        assert "tfidf" in names and "clf" in names

    def test_tfidf_config(self):
        p = build_pipeline(max_features=10000, C=0.5, max_iter=50)
        tfidf = p.named_steps["tfidf"]
        assert tfidf.max_features == 10000
        assert tfidf.ngram_range == (1, 2)

    def test_clf_config(self):
        p = build_pipeline(max_features=5000, C=2.0, max_iter=200)
        clf = p.named_steps["clf"]
        assert clf.C == 2.0
        assert clf.max_iter == 200


class TestPipelineTraining:
    @pytest.fixture
    def trained(self):
        p = build_pipeline(max_features=500, C=1.0, max_iter=100)
        p.fit(SAMPLE_REVIEWS, SAMPLE_LABELS)
        return p

    def test_predict_length(self, trained):
        assert len(trained.predict(SAMPLE_REVIEWS)) == len(SAMPLE_REVIEWS)

    def test_predictions_binary(self, trained):
        assert set(trained.predict(SAMPLE_REVIEWS)).issubset({0, 1})

    def test_proba_shape(self, trained):
        assert trained.predict_proba(SAMPLE_REVIEWS).shape == (len(SAMPLE_REVIEWS), 2)

    def test_proba_sum_to_one(self, trained):
        np.testing.assert_allclose(
            trained.predict_proba(SAMPLE_REVIEWS).sum(axis=1), 1.0, atol=1e-6
        )

    def test_positive_review(self, trained):
        assert trained.predict(["Best movie ever, absolutely loved every minute!"])[0] == 1

    def test_negative_review(self, trained):
        assert trained.predict(["Terrible, awful, worst film I have ever seen."])[0] == 0

class TestEvaluate:
    @pytest.fixture
    def trained(self):
        p = build_pipeline(max_features=500, C=1.0, max_iter=100)
        p.fit(SAMPLE_REVIEWS, SAMPLE_LABELS)
        return p

    def test_returns_dict(self, trained):
        assert isinstance(evaluate(trained, SAMPLE_REVIEWS, SAMPLE_LABELS), dict)

    def test_required_keys(self, trained):
        m = evaluate(trained, SAMPLE_REVIEWS, SAMPLE_LABELS)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in m

    def test_metrics_in_range(self, trained):
        m = evaluate(trained, SAMPLE_REVIEWS, SAMPLE_LABELS)
        for k, v in m.items():
            assert 0.0 <= v <= 1.0

class TestDataValidation:
    def test_same_length(self):
        assert len(SAMPLE_REVIEWS) == len(SAMPLE_LABELS)

    def test_binary_labels(self):
        assert all(l in {0, 1} for l in SAMPLE_LABELS)

    def test_non_empty_strings(self):
        for r in SAMPLE_REVIEWS:
            assert isinstance(r, str) and len(r.strip()) > 0

    def test_balanced_classes(self):
        assert sum(SAMPLE_LABELS) == len(SAMPLE_LABELS) - sum(SAMPLE_LABELS)