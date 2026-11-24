import os, warnings, pickle; warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, tensorflow as tf, holidays
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
A=pd.read_pickle(os.path.join(os.path.dirname(__file__),'A.pkl')); horas=pickle.load(open(os.path.join(os.path.dirname(__file__),'horas.pkl'),'rb'))
co=holidays.Colombia(years=range(2019,2025))
A['fest']=A['Día'].dt.date.map(lambda d: d in co).astype(int)
A['dow']=A['Día'].dt.dayofweek
A['y']=A['Any_Outlier_outlier'].astype(int)
X=A[horas].values.astype(float); y=A['y'].values
DOW=np.eye(7)[A['dow'].values]
def build_ae(d_in, d_out, seed):
    tf.keras.utils.set_random_seed(seed)
    inp=tf.keras.layers.Input(shape=(d_in,)); x=inp
    for u in (16,8,4,8,16): x=tf.keras.layers.Dense(u,activation='relu')(x)
    out=tf.keras.layers.Dense(d_out,activation='sigmoid')(x)
    m=tf.keras.models.Model(inp,out); m.compile(optimizer='adam',loss='mae'); return m
def fit_ae(Xtr_s, Xva_s, seed, cond_tr=None, cond_va=None):
    itr = Xtr_s if cond_tr is None else np.hstack([Xtr_s,cond_tr]); iva = Xva_s if cond_va is None else np.hstack([Xva_s,cond_va])
    m=build_ae(itr.shape[1], Xtr_s.shape[1], seed)
    m.fit(itr,Xtr_s,epochs=500,batch_size=32,validation_data=(iva,Xva_s),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=0)
    return m
class Scorers:
    """All scorers fitted ONLY on normal training days."""
    def __init__(self, tr_idx, va_norm_idx, seed):
        self.seed=seed; Xtr=X[tr_idx]
        self.sc=MinMaxScaler().fit(Xtr)
        self.ae=fit_ae(self.sc.transform(Xtr), self.sc.transform(X[va_norm_idx]), seed)
        self.aec=fit_ae(self.sc.transform(Xtr), self.sc.transform(X[va_norm_idx]), seed, DOW[tr_idx], DOW[va_norm_idx])
        rtr=self.sc.inverse_transform(self.ae.predict(self.sc.transform(Xtr),verbose=0))-Xtr
        self.res_sd=np.abs(rtr).mean(0)+1e-9
        rtrc=self.sc.inverse_transform(self.aec.predict(np.hstack([self.sc.transform(Xtr),DOW[tr_idx]]),verbose=0))-Xtr
        self.res_sdc=np.abs(rtrc).mean(0)+1e-9
        self.pca=PCA(4,random_state=seed).fit(self.sc.transform(Xtr))
        L=np.log1p(Xtr); d=A['dow'].values[tr_idx]
        self.med={k:np.median(L[d==k],0) for k in range(7)}; self.mad={k:np.median(np.abs(L[d==k]-self.med[k]),0)*1.4826+1e-6 for k in range(7)}
        self.iso=IsolationForest(n_estimators=300,random_state=seed).fit(np.hstack([self.sc.transform(Xtr),DOW[tr_idx]]))
    def scores(self, idx):
        Xi=X[idx]; Xs=self.sc.transform(Xi); d=A['dow'].values[idx]
        rec=self.sc.inverse_transform(self.ae.predict(Xs,verbose=0)); e=np.abs(rec-Xi)
        recc=self.sc.inverse_transform(self.aec.predict(np.hstack([Xs,DOW[idx]]),verbose=0)); ec=np.abs(recc-Xi)
        L=np.log1p(Xi); z=np.vstack([np.abs(L[i]-self.med[d[i]])/self.mad[d[i]] for i in range(len(idx))])
        pr=self.pca.inverse_transform(self.pca.transform(Xs))
        return {
          'AE original (MAE pasajeros)': e.mean(1),
          'AE, error normalizado por hora': (e/self.res_sd).mean(1),
          'AE + día semana, err. normalizado': (ec/self.res_sdc).mean(1),
          'PCA k=4 (MAE escalado)': np.abs(pr-Xs).mean(1),
          'IsolationForest (+día semana)': -self.iso.score_samples(np.hstack([Xs,DOW[idx]])),
          'Perfil robusto por día-semana (max|z|)': z[:,:19].max(1),
          'Solo calendario (festivo o diciembre)': (A['fest'].values[idx] | (A['Día'].dt.month.values[idx]==12)).astype(float),
        }
