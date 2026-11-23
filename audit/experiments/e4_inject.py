import json, joblib
from common import *
import common
from sklearn.model_selection import train_test_split
from tensorflow import keras
R='/home/user/AutoencoderMetroAnomalyApp/dev/results'
dep=keras.models.load_model(f'{R}/tf_modelo_77_81_78.keras'); dsc=joblib.load(f'{R}/scaler_77_81_78.pkl')
H={h[:2]:i for i,h in enumerate(horas)}
sun=np.median(X[(A.dow.values==6)&(y==0)],0)
def inj(Xb, kind, dows):
    Z=Xb.copy()
    if kind=='Reducción uniforme 30%': Z*=0.7
    elif kind=='Reducción uniforme 50%': Z*=0.5
    elif kind=='Cierre total 06-08h': Z[:,[H['06'],H['07'],H['08']]]=0
    elif kind=='Apertura tardía (04-05h = 0)': Z[:,[H['04'],H['05']]]=0
    elif kind=='Caída 50% pico 17-18h': Z[:,[H['17'],H['18']]]*=0.5
    elif kind=='Evento +50% 17-19h': Z[:,[H['17'],H['18'],H['19']]]*=1.5
    elif kind=='Perfil de domingo en día hábil': Z[:]=sun
    elif kind=='Desfase de 1 hora (error de registro)': Z=np.roll(Z,1,axis=1); Z[:,0]=Xb[:,0]
    elif kind=='Hora 10h duplicada (x2)': Z[:,H['10']]*=2
    elif kind=='Corte nocturno 20-22h = 0': Z[:,[H['20'],H['21'],H['22']]]=0
    return Z
kinds=['Reducción uniforme 30%','Reducción uniforme 50%','Cierre total 06-08h','Apertura tardía (04-05h = 0)','Caída 50% pico 17-18h','Evento +50% 17-19h','Perfil de domingo en día hábil','Desfase de 1 hora (error de registro)','Hora 10h duplicada (x2)','Corte nocturno 20-22h = 0']
idx=np.arange(len(A)); n=idx[y==0]
rows=[]
for seed in range(3):
    tr, rest = train_test_split(n, test_size=0.4, random_state=seed); nva, nte = train_test_split(rest, test_size=0.5, random_state=seed)
    S=Scorers(tr, nva, seed)
    wk=nte[A.dow.values[nte]<5]  # weekday clean test normals
    base=S.scores(wk)
    Xb=X[wk]
    def dep_score(Z): return np.abs(dsc.inverse_transform(dep.predict(dsc.transform(Z),verbose=0))-Z).mean(1)
    dep_base=dep_score(Xb)
    for kind in kinds:
        Z=inj(Xb,kind,None)
        # temporarily swap X rows to score injected copies
        Xorig=common.X[wk].copy(); common.X[wk]=Z; sc_=S.scores(wk); common.X[wk]=Xorig
        r=dict(seed=seed,kind=kind,deployed_827=(dep_score(Z)>=827).mean())
        r['deployed_827_FPR']=(dep_base>=827).mean()
        for k in base:
            if k.startswith('Solo calendario'): continue
            t=np.quantile(base[k],0.95)  # threshold @ 5% FPR on clean test normals
            r[k]=(sc_[k]>t).mean()
        rows.append(r)
D=pd.DataFrame(rows); D.to_pickle('inject.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(D.drop(columns='seed').groupby('kind',sort=False).mean().round(2).T.to_string())
