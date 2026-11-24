# Reproduce cells 5-13 of dev/model.ipynb exactly (HEAD), without writing into repo
import os, warnings; warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys; sys.path.insert(0, os.path.dirname(__file__))
from nb_funcs import *
REPO='/home/user/AutoencoderMetroAnomalyApp/dev'
def build(years=(2022,2023,2024), start='2022-01-01', end='2024-06-30', verbose=False):
    raw={}
    for y in range(2019,2025):
        d,h=cargar_y_limpiar_datos(f'{REPO}/data/Afluencia_Metro_{y}.xlsx'); raw[y]=(d,h)
    horas=raw[2024][1]
    la=pd.concat([filtrar_por_linea(raw[y][0]) for y in years]).sort_values(by='Día')
    full=generar_dataframe_completo(la, start=start, end=end)
    prom=calcular_promedio_estacional(full, horas)
    full=rellenar_faltantes(full, prom, horas)
    full['Línea de Servicio']='LINEA A'
    import io, contextlib
    f=io.StringIO()
    with contextlib.redirect_stdout(f):
        desc=eliminar_tendencia_estacionalidad(full, horas)
    if verbose: print(f.getvalue())
    out=detectar_outliers_en_residuos(desc, umbral=1.5, data=full)
    aligned_all=full.merge(out.add_suffix('_outlier'), on='Día', how='left').fillna(0)
    return raw, horas, la, full, desc, out, aligned_all
