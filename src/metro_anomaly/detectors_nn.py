"""Detectores basados en autoencoder (requieren TensorFlow: requirements-research.txt).

- ``AEOriginalDetector``: réplica **sin fuga** del enfoque original. MinMax sobre conteos crudos,
  arquitectura 16-8-4-8-16 con salida sigmoide, pérdida MAE y early stopping; score = MAE en pasajeros.
  Diferencia con el original: usa las 19 horas puntuadas (04–22), sin 23h (ver F2.3).
- ``AEContextDetector``: misma arquitectura, entrada = log1p - nivel (estandarizado) + tipo de día y rasgos
  de calendario; salida lineal; z por hora = residuo / MAD de entrenamiento.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .calendar import DAY_TYPES, calendar_features, day_type
from .data import SCORED_HOURS
from .detectors import MAD_K, SIGMA_FLOOR, Detector, Scores, aggregate

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _tf():
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def _build(tf, d_in, d_out, out_act, seed):
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.layers.Input(shape=(d_in,))
    x = inp
    for u in (16, 8, 4, 8, 16):
        x = tf.keras.layers.Dense(u, activation="relu")(x)
    out = tf.keras.layers.Dense(d_out, activation=out_act)(x)
    m = tf.keras.models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m


def _fit_time_split(tf, m, Xin, Yout, val_frac=0.15):
    n = len(Xin)
    k = max(int(n * (1 - val_frac)), 1)  # validación = últimos días (orden temporal, sin barajar entre tramos)
    m.fit(Xin[:k], Yout[:k], validation_data=(Xin[k:], Yout[k:]), epochs=500, batch_size=32, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    return m


class AEOriginalDetector(Detector):
    name = "ae_original"
    uses_z = False

    def __init__(self, agg="max", seed=0):
        super().__init__(agg)
        self.seed = seed

    def fit(self, hist, context):
        from sklearn.preprocessing import MinMaxScaler

        tf = _tf()
        X = hist[SCORED_HOURS].dropna(how="any").to_numpy(float)
        self.sc_ = MinMaxScaler().fit(X)
        Xs = self.sc_.transform(X)
        self.m_ = _fit_time_split(tf, _build(tf, Xs.shape[1], Xs.shape[1], "sigmoid", self.seed), Xs, Xs)
        return self

    def score(self, X, context):
        A = X[SCORED_HOURS].to_numpy(float)
        s = np.full(len(A), np.nan)
        ok = ~np.isnan(A).any(1)
        if ok.any():
            rec = self.sc_.inverse_transform(self.m_.predict(self.sc_.transform(A[ok]), verbose=0))
            s[ok] = np.abs(rec - A[ok]).mean(1)
        return Scores(pd.Series(s, index=X.index))


class AEContextDetector(Detector):
    name = "ae_contextual"

    def __init__(self, agg="max", seed=0):
        super().__init__(agg)
        self.seed = seed

    @staticmethod
    def _cal(dates):
        f = calendar_features(dates)[["festivo", "especial", "vispera_festivo", "post_festivo", "fin_de_anio",
                                      "sin_doy_1", "cos_doy_1"]]
        dt_ = day_type(dates)
        for t in DAY_TYPES:
            f[f"tipo_{t}"] = (dt_ == t).astype(float)
        return f.to_numpy()

    def fit(self, hist, context):
        tf = _tf()
        Y = self._deleveled(hist, context).dropna(how="any")
        self.mu_, self.sd_ = Y.mean().to_numpy(), Y.std().to_numpy() + 1e-6
        Ys = (Y.to_numpy() - self.mu_) / self.sd_
        Xin = np.hstack([Ys, self._cal(Y.index)])
        self.m_ = _fit_time_split(tf, _build(tf, Xin.shape[1], Ys.shape[1], "linear", self.seed), Xin, Ys)
        R = (self.m_.predict(Xin, verbose=0) - Ys) * self.sd_
        self.sig_ = np.clip(MAD_K * np.median(np.abs(R - np.median(R, 0)), 0), SIGMA_FLOOR, None)
        return self

    def score(self, X, context):
        Y = self._deleveled(X, context)
        z = pd.DataFrame(np.nan, index=X.index, columns=SCORED_HOURS)
        ok = Y.notna().all(axis=1).to_numpy()
        if ok.any():
            Ys = (Y.to_numpy()[ok] - self.mu_) / self.sd_
            rec = self.m_.predict(np.hstack([Ys, self._cal(X.index[ok])]), verbose=0)
            z.iloc[ok] = (Ys - rec) * self.sd_ / self.sig_
        return Scores(aggregate(z, self.agg), z)


REGISTRY_NN = {"ae_original": AEOriginalDetector, "ae_contextual": AEContextDetector}
