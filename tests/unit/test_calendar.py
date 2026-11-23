import numpy as np
import pandas as pd

from metro_anomaly import calendar as C


def test_fixed_and_moved_holidays():
    # 2023-01-01 (Año Nuevo, domingo); 2023-01-09 (Reyes trasladado a lunes); 2023-07-20 (fijo)
    d = pd.to_datetime(["2023-01-01", "2023-01-06", "2023-01-09", "2023-07-20", "2023-07-21"])
    assert C.is_holiday(d).tolist() == [True, False, True, True, False]


def test_easter_based_holidays_2024():
    # Semana Santa 2024: jueves 28/03 y viernes 29/03
    assert C.is_holiday(pd.to_datetime(["2024-03-28", "2024-03-29", "2024-03-27"])).tolist() == [True, True, False]


def test_day_type():
    d = pd.to_datetime(["2023-03-15", "2023-03-18", "2023-03-19", "2023-03-20", "2023-12-24", "2023-12-25", "2022-12-24"])
    # mié, sáb, dom, lun festivo (San José), 24/12/2023 domingo (especial), Navidad, 24/12/2022 sábado (especial)
    assert C.day_type(d).tolist() == ["habil", "sabado", "domingo_festivo", "domingo_festivo", "especial", "domingo_festivo", "especial"]


def test_calendar_features_shape_and_values():
    d = pd.date_range("2023-01-01", "2023-12-31")
    f = C.calendar_features(d)
    assert len(f) == 365 and f.filter(like="dow_").sum(axis=1).eq(1).all()
    assert f.loc["2023-01-08", "vispera_festivo"] == 1 and f.loc["2023-01-10", "post_festivo"] == 1
    assert np.isfinite(f.to_numpy()).all()


def test_holiday_name():
    assert C.holiday_name("2023-12-25") is not None and C.holiday_name("2023-12-26") is None
