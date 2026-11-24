"""Contraste independiente de las afirmaciones de audit/AUDITORIA.md.

Cada test reimplementa la verificación desde cero (sin reutilizar el código del notebook)
y se ejecuta sobre los artefactos versionados del repositorio. Si un test falla, la
afirmación de la auditoría queda REFUTADA y debe registrarse en docs/BITACORA.md.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "dev" / "results"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def labelled():
    df = pd.read_excel(RES / "full_2019_2024_data_linea_A_outliers.xlsx")
    hours = [c for c in df.columns if isinstance(c, str) and c.endswith(":00:00") and not c.endswith("_outlier")]
    df = df.sort_values("Día").reset_index(drop=True)
    return df, hours


def _iqr_flags(values: pd.Series, groups: pd.Series, k: float = 1.5) -> pd.Series:
    out = pd.Series(0, index=values.index)
    for _, idx in values.groupby(groups).groups.items():
        s = values.loc[idx]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out.loc[idx] = ((s < q1 - k * iqr) | (s > q3 + k * iqr)).astype(int)
    return out


def test_C1_labels_equal_iqr_on_raw_values_not_residuals(labelled):
    df, hours = labelled
    # los 3 primeros/últimos días no tienen residuo (media móvil centrada, periodo 7) -> se excluyen
    core = df.iloc[3:-3]
    dow = core["Día"].dt.dayofweek
    raw_any = pd.concat([_iqr_flags(core[h], dow) for h in hours[:-1]], axis=1).max(axis=1)
    assert (raw_any.values == core["Any_Outlier_outlier"].astype(int).values).all()

    resid_flags = []
    for h in hours[:-1]:
        r = seasonal_decompose(df.set_index("Día")[h], model="multiplicative", period=7).resid
        r = pd.Series(r.values, index=df.index).loc[core.index]
        resid_flags.append(_iqr_flags(r, dow))
    resid_any = pd.concat(resid_flags, axis=1).max(axis=1)
    disagreement = (resid_any.values != core["Any_Outlier_outlier"].astype(int).values).mean()
    assert disagreement > 0.10  # con residuos reales las etiquetas cambian sustancialmente


def test_A1_last_hour_never_evaluated(labelled):
    df, hours = labelled
    assert hours[-1] == "23:00:00"
    assert df["23:00:00_outlier"].sum() == 0
    assert all(df[f"{h}_outlier"].sum() > 0 for h in hours[:-1])


def test_A2_decomposition_selection_compares_incompatible_units(labelled):
    df, hours = labelled
    s = df.set_index("Día")
    for h in hours:
        add = seasonal_decompose(s[h], model="additive", period=7).resid.dropna()
        mul = seasonal_decompose(s[h], model="multiplicative", period=7).resid.dropna()
        mse_add, mse_mul = float(np.mean(add**2)), float(np.mean(mul**2))
        assert 0.5 < mse_mul < 3.0  # residuo multiplicativo centrado en 1
        assert mse_add > 100 * mse_mul  # la multiplicativa "gana" siempre por escala


def test_C2_deployed_scaler_fitted_on_all_912_days(labelled):
    df, hours = labelled
    sc = joblib.load(RES / "scaler_77_81_78.pkl")
    assert sc.n_samples_seen_ == 912 == len(df)
    np.testing.assert_allclose(sc.data_min_, df[hours].min().values)
    np.testing.assert_allclose(sc.data_max_, df[hours].max().values)


def test_C5_holidays_and_december_dominate_labels(labelled):
    import holidays

    df, _ = labelled
    co = holidays.Colombia(years=range(2022, 2025))
    fest = df["Día"].dt.date.map(lambda d: d in co)
    dec = df["Día"].dt.month == 12
    y = df["Any_Outlier_outlier"].astype(int)
    assert y[fest].mean() > 0.9
    assert y[dec].mean() > 0.9
    assert y[~fest & ~dec].mean() < 0.25
