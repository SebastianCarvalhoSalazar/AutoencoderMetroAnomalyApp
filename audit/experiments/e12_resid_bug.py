from pipeline import *
raw, horas, la, full, desc, out, A = build()
h='07:00:00'
r=desc[h].resid
print('resid series name:', r.name)
m=pd.DataFrame(r.dropna()).merge(full, on='Día', how='left')
print('columns after merge (first 5):', list(m.columns)[:5])
print('column used by code  [hora] equals RAW values?', np.allclose(m[h].values, full.set_index('Día').loc[m['Día'] if 'Día' in m else r.dropna().index, h].values))
# Recompute labels: (a) raw values per weekday IQR (what code does), (b) true residuals per weekday IQR (what is described)
def lab(use_resid):
    idx=out.index; M=pd.DataFrame(0,index=idx,columns=horas[:-1])
    F=full.set_index('Día').loc[idx]
    for hh in horas[:-1]:
        v = desc[hh].resid.dropna().loc[idx] if use_resid else F[hh]
        for d in F['Día de la Semana'].unique():
            s=v[F['Día de la Semana']==d]; q1,q3=s.quantile(.25),s.quantile(.75); i=q3-q1
            M.loc[s.index,hh]=((s<q1-1.5*i)|(s>q3+1.5*i)).astype(int)
    return (M.sum(1)>0).astype(int)
a=lab(False); b=lab(True); orig=out['Any_Outlier']
print('labels == raw-value IQR reimplementation:', (a==orig).mean(), '| labels == TRUE residual IQR:', (b==orig).mean())
print('prevalence code(raw)=%.3f  true-residual=%.3f'%(a.mean(), b.mean()))
print('agreement raw vs residual labels:', pd.crosstab(a.rename('raw'), b.rename('resid')).to_dict())
import holidays; co=holidays.Colombia(years=range(2022,2025))
fest=pd.Series([d.date() in co for d in out.index], index=out.index)
dec=pd.Series(out.index.month==12, index=out.index)
for nm,l in [('raw (current)',a),('true residual',b)]:
    print(nm,': holidays flagged %.2f | December flagged %.2f | other days flagged %.2f'%(l[fest].mean(), l[dec].mean(), l[~fest & ~dec].mean()))
b.to_pickle('labels_true_resid.pkl')
