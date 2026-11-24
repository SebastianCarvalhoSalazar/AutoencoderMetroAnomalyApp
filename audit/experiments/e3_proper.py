import sys, json
from common import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
def best_f1_thr(s,yv):
    c=np.unique(s); best=(-1,None)
    for t in c:
        f=f1_score(yv,(s>=t).astype(int)); best=max(best,(f,t))
    return best[1]
def evaluate(tr, va, te, seed, tag):
    S=Scorers(tr, va[y[va]==0], seed); sv=S.scores(va); st=S.scores(te); out=[]
    for k in sv:
        t=best_f1_thr(sv[k],y[va]); p=(st[k]>=t).astype(int)
        out.append(dict(split=tag,seed=seed,method=k,auc=roc_auc_score(y[te],st[k]),ap=average_precision_score(y[te],st[k]),f1=f1_score(y[te],p),prec=precision_score(y[te],p,zero_division=0),rec=recall_score(y[te],p),prev=y[te].mean()))
    return out
mode=sys.argv[1]; seeds=range(int(sys.argv[2]),int(sys.argv[3]))
idx=np.arange(len(A))
for seed in seeds:
    if mode=='random':
        n=idx[y==0]; o=idx[y==1]
        tr, rest = train_test_split(n, test_size=0.4, random_state=seed)
        nva, nte = train_test_split(rest, test_size=0.5, random_state=seed)
        ova, ote = train_test_split(o, test_size=0.5, random_state=seed)
        va=np.r_[nva,ova]; te=np.r_[nte,ote]
    else:  # temporal
        d=A['Día']
        trp=idx[(d<'2023-01-01').values]; tr=trp[y[trp]==0]
        va=idx[((d>='2023-01-01')&(d<'2023-07-01')).values]; te=idx[(d>='2023-07-01').values]
    for r in evaluate(tr,va,te,seed,mode): print(json.dumps(r),flush=True)
