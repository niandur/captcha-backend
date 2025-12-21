from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mysql.connector
import json
import traceback
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})
# --- CONFIGURACIÓN BASE DE DATOS ---
DB_CONFIG = {
    'host': 'qaot733.niandur.com',
    'user': 'qaot733',
    'password': 'Omega73bd',
    'database': 'qaot733'
}

# --- CARGA DE ARTEFACTOS (CEREBROS DEL MODELO) ---
# Asegúrate de que estos archivos estén en la misma carpeta
try:
    print("⏳ Cargando modelos...")
    MODELO = joblib.load('modelo_rf_final.pkl')
    SCALER = joblib.load('escalador_final.pkl')
    # La lista exacta de columnas que me diste:
    COLUMNAS_ENTRENAMIENTO = [
        'longitud_trayectoria', 'distancia_total_norm', 'velocidad_media_norm', 
        'velocidad_std', 'aceleracion_std', 'ratio_clics', 'curvatura_media', 
        'curvatura_std', 'tiempo_total', 'forma_estimativa_lineal', 'forma_estimativa_suave'
    ]
    print("✅ Modelos cargados. Sistema listo para inferencia.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELOS: {e}")
    print("El sistema funcionará en modo 'colección de datos' pero no podrá predecir.")
    MODELO = None
    SCALER = None
    COLUMNAS_ENTRENAMIENTO = []

def preprocesar_input(data_raw):
    """
    Transforma el JSON del frontend en un DataFrame idéntico al del entrenamiento.
    Replica EXACTAMENTE la lógica de 'calcular_features' del Notebook.
    """
    # 1. Extraer datos crudos
    points = data_raw.get('mouse_movements', [])
    num_clics = data_raw.get('clicks', 0) # Asegúrate de que el frontend envíe 'clicks'

    # Valores por defecto si la trayectoria está vacía
    features = {
        'longitud_trayectoria': 0,
        'distancia_total_norm': 0.0,
        'velocidad_media_norm': 0.0,
        'velocidad_std': 0.0,
        'aceleracion_std': 0.0,
        'ratio_clics': 0.0,
        'curvatura_media': 0.0,
        'curvatura_std': 0.0,
        'tiempo_total': 0.0,
        # Variables categóricas desglosadas (One-Hot Encoding manual)
        'forma_estimativa_lineal': 0,
        'forma_estimativa_suave': 0
    }

    if not points or len(points) < 2:
        return pd.DataFrame([features])

    # 2. Crear DataFrame de la trayectoria (Igual que en el Notebook)
    df_mov = pd.DataFrame(points)
    
    # Validar que vengan las columnas necesarias
    if not {'x', 'y', 't'}.issubset(df_mov.columns):
        return pd.DataFrame([features])

    # Ordenar y limpiar
    df_mov = df_mov.sort_values('t')
    
    # --- CÁLCULOS MATEMÁTICOS (Réplica del Notebook) ---
    df_mov['dt'] = df_mov['t'].diff()
    df_mov['dx'] = df_mov['x'].diff()
    df_mov['dy'] = df_mov['y'].diff()
    
    df_diff = df_mov.dropna().copy()
    df_diff['dt'] = df_diff['dt'].replace(0, 1e-6) # Evitar división por cero

    # Longitud
    longitud_trayectoria = len(points)
    
    # Distancia
    df_diff['dist_seg'] = np.sqrt(df_diff['dx']**2 + df_diff['dy']**2)
    distancia_total = df_diff['dist_seg'].sum()

    # Tiempo Total
    tiempo_total = df_mov['t'].max() - df_mov['t'].min()

    # Velocidad Media
    velocidad_media = distancia_total / tiempo_total if tiempo_total > 0 else 0

    # Velocidad Std
    df_diff['velocidad_inst'] = df_diff['dist_seg'] / df_diff['dt']
    velocidad_std = df_diff['velocidad_inst'].std()

    # Aceleración Std
    df_diff['dv'] = df_diff['velocidad_inst'].diff()
    df_diff['aceleracion_inst'] = df_diff['dv'] / df_diff['dt']
    aceleracion_std = df_diff['aceleracion_inst'].std()

    # Ratio Clics
    ratio_clics = num_clics / longitud_trayectoria if longitud_trayectoria > 0 else 0

    # Curvatura
    angulos = np.arctan2(df_diff['dy'], df_diff['dx'])
    diff_angulos = np.diff(angulos)
    diff_angulos = (diff_angulos + np.pi) % (2 * np.pi) - np.pi
    diff_angulos_abs = np.abs(diff_angulos)

    if len(diff_angulos_abs) > 0:
        curvatura_media = np.mean(diff_angulos_abs)
        curvatura_std = np.std(diff_angulos_abs)
    else:
        curvatura_media = 0.0
        curvatura_std = 0.0

    # Determinar Forma (Lógica original)
    forma = 'curva' # Valor por defecto (dropping category)
    if curvatura_media < 0.1 and curvatura_std < 0.2:
        forma = 'lineal'
    elif curvatura_media < 0.5:
        forma = 'suave'

    # Limpieza de NaNs
    velocidad_std = 0.0 if np.isnan(velocidad_std) else velocidad_std
    aceleracion_std = 0.0 if np.isnan(aceleracion_std) else aceleracion_std

    # --- ASIGNACIÓN AL DICCIONARIO FINAL ---
    features['longitud_trayectoria'] = int(longitud_trayectoria)
    features['distancia_total_norm'] = float(distancia_total) # Mantengo el nombre _norm para coincidir con tu entreno
    features['velocidad_media_norm'] = float(velocidad_media)
    features['velocidad_std'] = float(velocidad_std)
    features['aceleracion_std'] = float(aceleracion_std)
    features['ratio_clics'] = float(ratio_clics)
    features['curvatura_media'] = float(curvatura_media)
    features['curvatura_std'] = float(curvatura_std)
    features['tiempo_total'] = float(tiempo_total)
    
    # One-Hot Encoding Manual para coincidir con COLUMNAS_ENTRENAMIENTO
    features['forma_estimativa_lineal'] = 1 if forma == 'lineal' else 0
    features['forma_estimativa_suave'] = 1 if forma == 'suave' else 0
    
    # Crear DataFrame de 1 fila
    return pd.DataFrame([features])

def guardar_captcha(data, prediccion, probabilidad):
    """Guarda los datos y la predicción en MySQL"""
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
        INSERT INTO interacciones 
        (session_id, user_agent, timestamp, total_time, mouse_data, keystrokes_data, 
         es_bot_prediccion, probabilidad_bot, navegador_info)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Serializar JSONs
        mouse_json = json.dumps(data.get('mouse_movements', []))
        keys_json = json.dumps(data.get('keystrokes', []))
        nav_json = json.dumps(data.get('navigator_info', {}))
        
        valores = (
            data.get('session_id', 'unknown'),
            data.get('user_agent', 'unknown'),
            datetime.now(),
            data.get('total_time', 0),
            mouse_json,
            keys_json,
            int(prediccion),
            float(probabilidad),
            nav_json
        )
        cursor.execute(query, valores)
        conn.commit()
        print(f"💾 Guardado en BD. ID: {cursor.lastrowid}")
    except Exception as e:
        print(f"❌ Error guardando en BD: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/verify-human', methods=['POST'])
def verify_human():
    data = request.json
    
    prediccion = 0 # 0 = Humano (por defecto)
    probabilidad_bot = 0.0
    
    if MODELO and SCALER:
        try:
            # 1. Calcular Features (Igual que el notebook)
            df_features = preprocesar_input(data)
            
            # 2. Asegurar orden de columnas
            # Esto filtra cualquier feature extra y pone el orden correcto
            # Si falta alguna columna crítica, rellenamos con 0
            for col in COLUMNAS_ENTRENAMIENTO:
                if col not in df_features.columns:
                    df_features[col] = 0
            
            df_final = df_features[COLUMNAS_ENTRENAMIENTO]
            
            # 3. Escalar (Usando el cerebro guardado)
            X_scaled = SCALER.transform(df_final)
            
            # 4. Predecir
            prediccion = MODELO.predict(X_scaled)[0]
            probabilidad_bot = MODELO.predict_proba(X_scaled)[0][1]
            
            print(f"🧠 Análisis: {'🤖 BOT' if prediccion == 1 else '👤 HUMANO'}")
            print(f"📊 Probabilidad de ser Bot: {probabilidad_bot:.4f}")
            print(f"📉 Datos procesados: {df_final.iloc[0].to_dict()}")

        except Exception as e:
            print(f"⚠️ Error durante la predicción: {e}")
            # En caso de error, dejamos pasar (fail-open) o bloqueamos (fail-closed) según prefieras
            
    # Guardar en BD
    guardar_captcha(data, prediccion, probabilidad_bot)
    
    # Lógica de respuesta (Umbral de decisión)
    # Si el modelo dice Bot (1), devolvemos is_human = False
    es_humano = True if prediccion == 0 else False
    
    return jsonify({
        "success": True,
        "is_human": es_humano,
        "bot_probability": probabilidad_bot
    })

if __name__ == '__main__':
    print("🚀 Iniciando Backend Captcha V2...")
    app.run(host='0.0.0.0', port=5000, debug=True)