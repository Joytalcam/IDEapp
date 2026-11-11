from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
import os
import threading
import webbrowser
from datetime import datetime
import json

app = Flask(__name__)

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Rutas del modelo y scaler
MODEL_PATH = "Modelo/Modelo.joblib"
SCALER_PATH = "Modelo/scaler_X.joblib"

# Ruta del historial: usa DATA_DIR si existe (para Docker), si no, la raíz
DATA_DIR = os.environ.get('DATA_DIR', '.')
os.makedirs(DATA_DIR, exist_ok=True)  # Crear carpeta si no existe
HISTORIAL_FILE = os.path.join(DATA_DIR, "historial_predicciones.json")

# Features del modelo (mismo orden que en entrenamiento)
FEATURES = ['Aves Ingresadas', 'Peso Promedio', 'Horas Productivas', 'LbsTotEquiv']

# Parámetros de la ecuación de línea base: IDE = A * lbs^B
A_BASELINE = 582.6204656920573
B_BASELINE = -0.713698689160424

# Capacidad nominal de la planta (aves procesadas por hora)
# Fuente: [COMPLETAR - ej: ficha técnica de la línea, promedio histórico, etc.]
CAPACIDAD_NOMINAL_AVES_HORA = 14000

# Límite máximo de registros en historial (evita crecimiento indefinido)
MAX_HISTORIAL = 1000

# Flag de debug (activar con: $env:DEBUG_PREDICCION="true" antes de correr)
DEBUG_PREDICCION = os.environ.get('DEBUG_PREDICCION', 'false').lower() == 'true'

# ============================================================
# ESTADO GLOBAL
# ============================================================

historial_predicciones = []
modelo_cargado = None
scaler_cargado = None


# ============================================================
# UTILIDADES
# ============================================================

def debug_print(*args, **kwargs):
    """Imprime solo si DEBUG_PREDICCION está activo"""
    if DEBUG_PREDICCION:
        print(*args, **kwargs)


# ============================================================
# CARGA DE MODELO Y SCALER
# ============================================================

def cargar_modelo():
    """Carga el modelo (cacheado en memoria global)"""
    global modelo_cargado
    if modelo_cargado is None:
        try:
            modelo_cargado = joblib.load(MODEL_PATH)
            print(f"✓ Modelo cargado desde {MODEL_PATH}")
        except FileNotFoundError:
            raise Exception(f"No se encontró el modelo en: {MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {str(e)}")
    return modelo_cargado


def cargar_scaler():
    """Carga el scaler (cacheado en memoria global)"""
    global scaler_cargado
    if scaler_cargado is None:
        try:
            scaler_cargado = joblib.load(SCALER_PATH)
            print(f"✓ Scaler cargado desde {SCALER_PATH}")
        except FileNotFoundError:
            raise Exception(f"No se encontró el scaler en: {SCALER_PATH}")
        except Exception as e:
            raise Exception(f"Error al cargar el scaler: {str(e)}")
    return scaler_cargado


# ============================================================
# CÁLCULOS DE NEGOCIO
# ============================================================

def calcular_ide_baseline(lbs_tot_equiv):
    """Calcula el IDE de línea base con la ecuación de potencia"""
    return A_BASELINE * (lbs_tot_equiv ** B_BASELINE)


def calcular_consumo_energetico(ide_modelo, lbs_tot_equiv):
    """Estima el consumo en kWh multiplicando IDE por libras procesadas"""
    kwh = ide_modelo * lbs_tot_equiv
    if kwh < 0:
        print(f"⚠️  ADVERTENCIA: consumo negativo ({kwh:.2f} kWh) — "
              f"posible predicción fuera de rango del modelo")
        return 0
    return kwh


def calcular_eficiencia_operacional(aves_ingresadas, horas_productivas):
    """
    Calcula la eficiencia operacional como aves/hora normalizadas
    contra la capacidad nominal de la planta.
    
    Retorna un valor entre 0 y 1.0 (donde 1.0 = 100% de capacidad).
    """
    if horas_productivas == 0:
        return 0
    
    tasa_real = aves_ingresadas / horas_productivas
    eficiencia = tasa_real / CAPACIDAD_NOMINAL_AVES_HORA
    
    if eficiencia > 1.0:
        print(f"⚠️  Eficiencia {eficiencia:.2f} excede capacidad nominal "
              f"({tasa_real:.0f} aves/h vs {CAPACIDAD_NOMINAL_AVES_HORA} nominal)")
        return 1.0
    
    return eficiencia


def calcular_desviacion_ide(ide_modelo, ide_baseline):
    """Calcula la desviación porcentual del IDE respecto a la línea base"""
    if ide_baseline == 0:
        return 0
    return ((ide_modelo - ide_baseline) / ide_baseline) * 100


# ============================================================
# PERSISTENCIA DEL HISTORIAL
# ============================================================

def guardar_historial():
    """Persiste el historial con escritura atómica (evita archivos corruptos)"""
    try:
        temp_file = HISTORIAL_FILE + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(historial_predicciones, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, HISTORIAL_FILE)  # Atómico en Windows y Unix
    except Exception as e:
        print(f"❌ Error al guardar historial: {e}")


def cargar_historial():
    """Recupera el historial guardado, manejando JSON corrupto con backup"""
    global historial_predicciones
    try:
        if not os.path.exists(HISTORIAL_FILE):
            print("ℹ️  No hay historial previo, empezando desde cero")
            return
        
        with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        if isinstance(datos, list):
            historial_predicciones = datos
            print(f"✓ Historial cargado: {len(datos)} registros")
        else:
            print("⚠️  Historial con formato inválido, se reinicia vacío")
            historial_predicciones = []
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON corrupto en {HISTORIAL_FILE}: {e}")
        backup = HISTORIAL_FILE + ".corrupto"
        try:
            os.rename(HISTORIAL_FILE, backup)
            print(f"   Archivo respaldado como: {backup}")
        except Exception as rename_err:
            print(f"   No se pudo respaldar: {rename_err}")
        historial_predicciones = []
    except Exception as e:
        print(f"❌ Error al cargar historial: {e}")
        historial_predicciones = []


# ============================================================
# RUTAS
# ============================================================

@app.route('/')
def index():
    """Mostrar el formulario principal"""
    return render_template('index.html', features=FEATURES)


@app.route('/predecir', methods=['POST'])
def predecir():
    """Procesar la predicción con todos los cálculos del IDE y métricas"""
    try:
        valores = []
        valores_dict = {}
        
        debug_print("\n\n" + "█" * 60)
        debug_print("█ DEBUG - NUEVA PREDICCIÓN")
        debug_print("█" * 60)
        
        # 1. Datos del formulario
        debug_print("\n1. DATOS DEL FORMULARIO:")
        for key in request.form:
            debug_print(f"     '{key}' = '{request.form.get(key)}'")
        
        # 2. Validar y procesar features
        debug_print("\n2. PROCESANDO FEATURES:")
        for feature in FEATURES:
            valor = request.form.get(feature)
            debug_print(f"     '{feature}' -> {valor}", end="")
            
            if not valor:
                return render_template('index.html',
                                     features=FEATURES,
                                     error=f"El campo '{feature}' es obligatorio")
            
            try:
                valor_float = float(valor)
                if valor_float <= 0:
                    return render_template('index.html',
                                         features=FEATURES,
                                         error=f"El campo '{feature}' debe ser mayor a cero")
                valores.append(valor_float)
                valores_dict[feature] = valor_float
                debug_print(f" ✓ ({valor_float})")
            except ValueError:
                return render_template('index.html',
                                     features=FEATURES,
                                     error=f"El campo '{feature}' debe ser un número válido")
        
        # 3. Preparar DataFrame
        df_entrada = pd.DataFrame([valores], columns=FEATURES)
        debug_print("\n3. DATAFRAME DE ENTRADA:")
        debug_print(df_entrada)
        
        # 4. Aplicar scaler
        scaler_X = cargar_scaler()
        datos_escalados_array = scaler_X.transform(df_entrada)
        df_escalado = pd.DataFrame(datos_escalados_array, columns=FEATURES)
        debug_print("\n4. DATOS ESCALADOS:")
        debug_print(df_escalado)
        
        # 5. Predecir
        modelo = cargar_modelo()
        ide_modelo = modelo.predict(df_escalado)[0]
        debug_print(f"\n5. IDE MODELO: {ide_modelo}")
        
        # 6. Cálculos adicionales
        lbs_tot_equiv = valores_dict['LbsTotEquiv']
        aves_ingresadas = valores_dict['Aves Ingresadas']
        horas_productivas = valores_dict['Horas Productivas']
        
        ide_baseline = calcular_ide_baseline(lbs_tot_equiv)
        kwh = calcular_consumo_energetico(ide_modelo, lbs_tot_equiv)
        eficiencia_op = calcular_eficiencia_operacional(aves_ingresadas, horas_productivas)
        desviacion_ide = calcular_desviacion_ide(ide_modelo, ide_baseline)
        
        debug_print(f"   IDE Baseline: {ide_baseline:.4f}")
        debug_print(f"   Consumo (kWh): {kwh:.4f}")
        debug_print(f"   Eficiencia op: {eficiencia_op:.4f}")
        debug_print(f"   Desviación IDE: {desviacion_ide:.2f}%")
        debug_print("=" * 60 + "\n")
        
        # 7. Guardar en historial
        registro = {
            'fecha_hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lbs_tot_equiv': float(lbs_tot_equiv),
            'ide_modelo': float(ide_modelo),
            'ide_baseline': float(ide_baseline),
            'kwh': float(kwh),
            'eficiencia_operacional': float(eficiencia_op),
            'desviacion_ide': float(desviacion_ide),
            'datos_entrada': valores_dict
        }
        
        historial_predicciones.append(registro)
        if len(historial_predicciones) > MAX_HISTORIAL:
            historial_predicciones.pop(0)
        guardar_historial()
        
        # 8. Preparar respuesta
        resultados = {
            'ide_modelo': float(ide_modelo),
            'ide_baseline': float(ide_baseline),
            'kwh': float(kwh),
            'eficiencia_op': float(eficiencia_op),
            'desviacion_ide': float(desviacion_ide),
            'desviacion_tipo': 'mejor' if desviacion_ide < 0 else 'peor'
        }
        
        return render_template('resultado.html',
                             resultados=resultados,
                             datos_entrada=valores_dict,
                             historial=historial_predicciones)
    
    except Exception as e:
        print(f"\n❌ ERROR EN PREDICCIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                             features=FEATURES,
                             error=f"Error al realizar la predicción: {str(e)}")


@app.route('/api/historial')
def api_historial():
    """Devolver el historial en JSON (para consumo de APIs)"""
    try:
        return jsonify({
            'success': True,
            'total': len(historial_predicciones),
            'data': historial_predicciones
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/historial')
def ver_historial():
    """Ver todo el historial de predicciones en la web"""
    return render_template('historial.html', historial=historial_predicciones)


@app.route('/limpiar_historial', methods=['POST'])
def limpiar_historial():
    """Borrar todo el historial cuando lo necesites"""
    global historial_predicciones
    historial_predicciones = []
    guardar_historial()
    return render_template('index.html',
                         features=FEATURES,
                         mensaje="Historial limpiado correctamente")


# ============================================================
# ARRANQUE
# ============================================================

if __name__ == '__main__':
    # Cargar historial previo
    cargar_historial()
    
    # Banner de arranque
    print("\n" + "=" * 60)
    print("  SISTEMA DE MONITOREO ENERGÉTICO INDUSTRIAL")
    print("  Predicción IDE - Análisis de Desempeño")
    print("=" * 60)
    
    # Verificar modelo y scaler
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No se encuentra el modelo en: {MODEL_PATH}")
    else:
        print(f"✓ Modelo encontrado: {MODEL_PATH}")
    
    if not os.path.exists(SCALER_PATH):
        print(f"❌ ERROR: No se encuentra el scaler en: {SCALER_PATH}")
    else:
        print(f"✓ Scaler encontrado: {SCALER_PATH}")
    
    print(f"\n  Features: {FEATURES}")
    print(f"  Debug: {'ACTIVO' if DEBUG_PREDICCION else 'inactivo'}")
    print(f"  URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    # Cargar modelo y scaler
    try:
        cargar_modelo()
        cargar_scaler()
    except Exception as e:
        print(f"❌ Error al cargar modelo/scaler: {e}")
    
    # Abrir navegador solo en uso local (no en Docker)
    # En Docker no hay navegador, así que saltamos esta parte
    EN_DOCKER = os.environ.get('DATA_DIR') is not None
    if not EN_DOCKER and os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    
    # Iniciar servidor (host y debug dinámicos según entorno)
    host = '0.0.0.0' if EN_DOCKER else '127.0.0.1'
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug_mode, host=host, port=5000)