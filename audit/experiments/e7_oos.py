from pipeline import *
import joblib
from tensorflow import keras
from sklearn.metrics import roc_auc_score
R=f'{REPO}/results'
dep=keras.models.load_model(f'{R}/tf_modelo_77_81_78.keras', compile=False); sc=joblib.load(f'{R}/scaler_77_81_78.pkl')
for yr in (2019,2021):
    raw,horas,la,full,desc,out,A=build(years=(yr,),start=f'{yr}-01-01',end=f'{yr}-12-31')
    Xo=A[horas].values.astype(float); yo=A['Any_Outlier_outlier'].values.astype(int)
    Xs=sc.transform(Xo); e=np.abs(sc.inverse_transform(dep.predict(Xs,verbose=0))-Xo).mean(1)
    print(yr,'days',len(A),'label prevalence %.2f'%yo.mean(),'| scaled values >1: %.1f%% of cells, <0: %.1f%%'%(100*(Xs>1).mean(),100*(Xs<0).mean()),
          '| flagged @827: %.2f'%(e>=827).mean(), '| FPR %.2f TPR %.2f'%((e[yo==0]>=827).mean(),(e[yo==1]>=827).mean()), '| AUC %.3f'%roc_auc_score(yo,e))
la=pd.read_pickle('la.pkl')
raw,horas,_,_,_,_,_=build()
tot={y: filtrar_por_linea(raw[y][0])[horas].sum(1).mean() for y in raw}
print('Mean daily Line-A entries by year:', {k:int(v) for k,v in tot.items()})
