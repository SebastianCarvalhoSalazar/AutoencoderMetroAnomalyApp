import json

import pandas as pd
import pytest

from metro_anomaly.artifacts import load_profile, save_profile
from metro_anomaly.detectors import ProfileContextDetector, RidgeCalendarDetector


def test_roundtrip_scores_identical(synth, tmp_path):
    det = ProfileContextDetector(agg="peak_shift").fit(synth.loc[:"2023-01-31"], synth)
    p = save_profile(det, 7.5, tmp_path / "m.json", {"line": "LINEA A"})
    det2, thr, doc = load_profile(p)
    days = synth.loc["2023-02-01":"2023-02-28"]
    pd.testing.assert_series_equal(det.score(days, synth).day, det2.score(days, synth).day)
    assert thr == 7.5 and doc["line"] == "LINEA A" and doc["detector"] == "perfil_contextual"


def test_rejects_non_profile(synth, tmp_path):
    with pytest.raises(TypeError):
        save_profile(RidgeCalendarDetector().fit(synth.loc[:"2023-01-31"], synth), 1.0, tmp_path / "x.json", {})


def test_rejects_bad_schema(synth, tmp_path):
    det = ProfileContextDetector().fit(synth.loc[:"2023-01-31"], synth)
    p = save_profile(det, 1.0, tmp_path / "m.json", {})
    d = json.loads(p.read_text()); d["schema_version"] = 99; p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_profile(p)
