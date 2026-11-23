# Replicates cells 16-23 of the notebook EXACTLY (incl. leakage) over many seeds
import os, warnings, pickle, sys, json; warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nb_funcs import entrenar_autoencoder, evaluar_autoencoder
A=pd.read_pickle('A.pkl'); horas=pickle.load(open('horas.pkl','rb'))
seeds=range(int(sys.argv[1]), int(sys.argv[2])); res=[]
for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    normales=A[A['Any_Outlier_outlier']==0][horas]; outliers=A[A['Any_Outlier_outlier']==1][horas]
    train_normales, restantes = train_test_split(normales, test_size=0.3, random_state=seed)
    outliers_test_val, outliers_test = train_test_split(outliers, test_size=0.5, random_state=seed)
    normales_test_val, normales_test = train_test_split(restantes, test_size=0.5, random_state=seed)
    val=pd.concat([outliers_test_val.assign(y=1), normales_test_val.assign(y=0)])
    scaler=MinMaxScaler(); Xtr=scaler.fit_transform(A[horas])  # as in notebook: ALL rows
    Xte=scaler.transform(normales_test); Xval=scaler.transform(val[horas])
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ae,h=entrenar_autoencoder(Xtr, Xte, 20) if False else (None,None)
    # same as entrenar_autoencoder but verbose=0
    inp=tf.keras.layers.Input(shape=(20,)); x=tf.keras.layers.Dense(16,activation='relu')(inp); x=tf.keras.layers.Dense(8,activation='relu')(x); x=tf.keras.layers.Dense(4,activation='relu')(x)
    x=tf.keras.layers.Dense(8,activation='relu')(x); x=tf.keras.layers.Dense(16,activation='relu')(x); out=tf.keras.layers.Dense(20,activation='sigmoid')(x)
    ae=tf.keras.models.Model(inp,out); ae.compile(optimizer='adam',loss='mae')
    h=ae.fit(Xtr,Xtr,epochs=500,batch_size=32,validation_data=(Xte,Xte),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=0)
    y=val['y'].values
    pred_n=ae.predict(Xval,verbose=0); errors=np.mean(np.abs(val[horas].values-scaler.inverse_transform(pred_n)),axis=1)
    thr=errors[y==0].mean()+0.5*errors[y==0].std()
    p=(errors>thr).astype(int)
    r=dict(seed=seed,epochs=len(h.history['loss']),thr=thr,acc=accuracy_score(y,p),prec=precision_score(y,p),rec=recall_score(y,p),f1=f1_score(y,p))
    # same model, PROPER-ish check: train only on train_normales instead
    res.append(r); print(json.dumps(r),flush=True)
