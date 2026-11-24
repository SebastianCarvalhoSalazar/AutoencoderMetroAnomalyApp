import os, warnings, glob, pickle, re; warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, joblib
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix
R='/home/user/AutoencoderMetroAnomalyApp/dev/results'
A=pd.read_pickle('A.pkl'); horas=pickle.load(open('horas.pkl','rb'))
m=keras.models.load_model(f'{R}/tf_modelo_77_81_78.keras'); sc=joblib.load(f'{R}/scaler_77_81_78.pkl')
mh=keras.models.load_model(f'{R}/tf_modelo_77_81_78.h5', compile=False)
X=A[horas].values.astype(float); y=A['Any_Outlier_outlier'].values.astype(int)
print('scaler data_min==min of ALL 912 rows:', np.allclose(sc.data_min_, X.min(0)), ' data_max==max ALL:', np.allclose(sc.data_max_, X.max(0)), ' n_samples_seen:', sc.n_samples_seen_)
print('scaler feature names:', list(sc.feature_names_in_)[:3], '...')
Xs=sc.transform(X)
print('.h5 and .keras identical predictions:', np.allclose(m.predict(Xs,verbose=0), mh.predict(Xs,verbose=0)))
rec=sc.inverse_transform(m.predict(Xs,verbose=0)); err=np.abs(rec-X).mean(1)
print('Full dataset (912 days, all were in the training set): AUC=%.3f'%roc_auc_score(y,err))
pred=(err>=827).astype(int)
tn,fp,fn,tp=confusion_matrix(y,pred).ravel(); print('threshold 827 on all 912 days: TN=%d FP=%d FN=%d TP=%d  acc=%.3f prec=%.3f rec=%.3f'%(tn,fp,fn,tp,(tn+tp)/len(y),tp/(tp+fp),tp/(tp+fn)))
print('FPR on normal days: %.3f'%(fp/(fp+tn)))
# per-hour contribution to MAE
contrib=np.abs(rec-X).mean(0); print('share of MAE from each hour (%):', dict(zip([h[:2] for h in horas], (100*contrib/contrib.sum()).round(1))))
print('mean flow per hour:', dict(zip([h[:2] for h in horas], X.mean(0).round(0).astype(int))))
# demo files
rows=[]
for f in glob.glob('/home/user/AutoencoderMetroAnomalyApp/dev/data_test/*.pkl'):
    v=pickle.load(open(f,'rb')); lab=1 if 'outlier' in f else 0; idx=int(re.findall(r'_(\d+)\.pkl',f)[0])
    raw=sc.inverse_transform(v.reshape(1,-1))[0]
    d=np.abs(X-raw).max(1); j=d.argmin()
    r=sc.inverse_transform(m.predict(v.reshape(1,-1),verbose=0)); e=np.abs(r-raw.reshape(1,-1)).mean()
    rows.append(dict(f=os.path.basename(f),lab=lab,idx=idx,match_err=d[j],date=A['Día'].iloc[j],lab_true=y[j],err=e,pred_anom=int(not(e<827)), vmin=v.min(), vmax=v.max(), dtype=str(v.dtype), shape=v.shape))
D=pd.DataFrame(rows)
D['exact']=D.match_err<1e-6
print('demo files:',len(D),' exact match to a row of the deployed dataset with deployed scaler:', D.exact.sum())
print(D.groupby(['lab','exact']).idx.agg(['count','min','max']))
E=D[D.exact]; print('label in filename == label in dataset (exact ones):', (E.lab==E.lab_true).mean())
for name,G in [('ALL 389 files',D),('exact-match files',E)]:
    tn,fp,fn,tp=confusion_matrix(G.lab,G.pred_anom).ravel(); print(name,': TN=%d FP=%d FN=%d TP=%d acc=%.3f'%(tn,fp,fn,tp,(tn+tp)/len(G)))
print('non-exact normal files: values outside [0,1]?', ((D[~D.exact].vmin<0)|(D[~D.exact].vmax>1)).sum(), ' median match err (passengers):', D[~D.exact].match_err.median())
D.to_pickle('demo.pkl')
