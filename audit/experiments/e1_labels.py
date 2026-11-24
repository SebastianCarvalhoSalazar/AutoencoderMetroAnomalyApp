from pipeline import *
raw, horas, la, full, desc, out, A = build(verbose=False)
print('horas', len(horas), horas[0], horas[-1])
print('counts Any_Outlier:', A['Any_Outlier_outlier'].value_counts().to_dict())
# compare to repo excel
ref=pd.read_excel(f'{REPO}/results/full_2019_2024_data_linea_A_outliers.xlsx')
print('ref counts', ref['Any_Outlier_outlier'].value_counts().to_dict())
cols=[c for c in ref.columns if c.endswith('_outlier')]
print('labels identical to repo excel:', (ref[cols].values==A[cols].values).all(), ' values identical:', np.allclose(ref[horas].values, A[horas].values))
# per-hour outlier rates
print('per-hour flagged rate:'); print((A[[h+'_outlier' for h in horas]].mean()*100).round(1).to_string())
# rows with no residual (edge days)
print('days in outlier matrix:', len(out), 'of', len(full), '-> days defaulted to 0 by fillna:', sorted(set(full['Día'])-set(out.index))[:3], '...', sorted(set(full['Día'])-set(out.index))[-3:])
# Why multiplicative wins: compare MSE magnitudes
for h in ['04:00:00','07:00:00','23:00:00']:
    s=full.set_index('Día')[h]
    a=seasonal_decompose(s,model='additive',period=7).resid.dropna()
    m=seasonal_decompose(s,model='multiplicative',period=7).resid.dropna()
    st=STL(s,period=7).fit().resid
    print(h,'MSE add=%.3g mult=%.3g stl=%.3g | mean mult resid=%.3f; mult-resid MSE about 1 (=mean^2+var): %.3f'%(np.mean(a**2),np.mean(m**2),np.mean(st**2),m.mean(), m.mean()**2+m.var(ddof=0)))
# missing data & imputations
print('Line A rows per year:', {y:len(filtrar_por_linea(raw[y][0])) for y in raw})
print('NaN per hour in raw line A 2022-2024H1 (before imputation):'); print(la[horas].isna().sum()[lambda x:x>0].to_dict())
print('Missing calendar days (fully imputed):', sorted(set(pd.date_range('2022-01-01','2024-06-30'))-set(la['Día'])))
print('duplicates (Día) in line A:', la['Día'].duplicated().sum())
print('Lines available 2024:', raw[2024][0]['Línea de Servicio'].unique().tolist())
print('Lines available 2019:', raw[2019][0]['Línea de Servicio'].unique().tolist())
print('date ranges:', {y:(str(raw[y][0]['Día'].min().date()),str(raw[y][0]['Día'].max().date())) for y in raw})
A.to_pickle('A.pkl'); full.to_pickle('full.pkl'); la.to_pickle('la.pkl')
import pickle; pickle.dump(horas,open('horas.pkl','wb'))
