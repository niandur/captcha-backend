from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mysql.connector
import json
import traceback
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# --- 1. CARGA DEL MODELO ---
# Asegúrate de subir 'modelo_bot_final.pkl' a la misma carpeta
try:
    MODELO = joblib.load('modelo_bot_final.pkl')
    print("✅ Modelo cargado correctamente.")
except:
    print("⚠️ ADVERTENCIA: No se encontró 'modelo_bot_final.pkl'. La predicción fallará.")
    MODELO = None

# --- 2. CONFIGURACIÓN BASE DE DATOS (INTACTA) ---
def get_connection():
    return mysql.connector.connect(
        host="qaot733.niandur.com",
        user="qaot733",
        password="Omega73bd!",
        database="qaot733"
    )

# --- 3. FUNCIÓN PARA EXTRAER FEATURES (NUEVA) ---
def calcular_features(puntos):
    if not puntos or len(puntos) < 2:
        # Devuelve un dataframe con ceros si no hay datos suficientes
        return pd.DataFrame([np.zeros(11)], columns=[
            'longitud_trayectoria', 'distancia_total_norm', 'velocidad_media_norm',
            'velocidad_std', 'aceleracion_std', 'ratio_clics', 'curvatura_media',
            'curvatura_std', 'tiempo_total', 'forma_estimativa_lineal', 'forma_estimativa_suave'
        ])

    df = pd.DataFrame(puntos, columns=['x', 'y', 't'])
    
    # Preprocesamiento seguro
    dx = df['x'].diff().fillna(0)
    dy = df['y'].diff().fillna(0)
    dt = df['t'].diff().fillna(0)
    
    distancias = np.sqrt(dx**2 + dy**2)
    # Evitar división por cero sumando un epsilon
    velocidades = distancias / (dt + 1e-9) 
    aceleraciones = velocidades.diff().fillna(0) / (dt + 1e-9)
    
    # 11 Features exactas del entrenamiento
    longitud_trayectoria = distancias.sum()
    distancia_total_norm = longitud_trayectoria # Asumiendo escalado similar
    velocidad_media_norm = velocidades.mean()
    velocidad_std = velocidades.std()
    aceleracion_std = aceleraciones.std()
    ratio_clics = 0 # Valor por defecto si no se calcula en frontend
    
    # Curvatura
    angulos = np.arctan2(dy, dx)
    cambio_angulo = np.diff(angulos)
    # Manejo de arrays vacíos para curvatura
    if len(cambio_angulo) > 0:
        curvatura_media = np.mean(np.abs(cambio_angulo))
        curvatura_std = np.std(cambio_angulo)
    else:
        curvatura_media = 0
        curvatura_std = 0
        
    tiempo_total = df['t'].iloc[-1] - df['t'].iloc[0] if len(df) > 0 else 0
    
    forma_estimativa_lineal = 1.0 if curvatura_media < 0.1 else 0.0
    forma_estimativa_suave = 1.0 if (curvatura_media >= 0.1 and curvatura_media < 0.5) else 0.0
    
    features = pd.DataFrame([{
        'longitud_trayectoria': longitud_trayectoria,
        'distancia_total_norm': distancia_total_norm,
        'velocidad_media_norm': velocidad_media_norm,
        'velocidad_std': velocidad_std,
        'aceleracion_std': aceleracion_std,
        'ratio_clics': ratio_clics,
        'curvatura_media': curvatura_media,
        'curvatura_std': curvatura_std,
        'tiempo_total': tiempo_total,
        'forma_estimativa_lineal': forma_estimativa_lineal,
        'forma_estimativa_suave': forma_estimativa_suave
    }])
    
    return features

# --- 4. RUTAS ---

@app.route("/guardar_captcha", methods=["POST", "OPTIONS"])
def guardar_captcha():
    if request.method == "OPTIONS":
        return "", 204

    datos = request.get_json()
    print("🧾 Recibido datos de:", datos.get("nombre"))

    try:
        # A) LÓGICA DE PREDICCIÓN (NUEVA)
        movimientos = datos.get("movimientos") or []
        X_input = calcular_features(movimientos)
        
        # Predicción: 0=Humano, 1=Bot (según tu entrenamiento)
        # Nota: Ajusta esto si tu etiqueta 1 es Humano. 
        # En tu árbol: class 1 solía ser Bot.
        es_bot = int(MODELO.predict(X_input)[0]) if MODELO else 0
        probabilidad = float(MODELO.predict_proba(X_input)[0][1]) if MODELO else 0.0
        
        etiqueta_predicha = "BOT" if es_bot == 1 else "HUMANO"

        # B) GUARDADO EN MYSQL (EXISTENTE + PREDICCIÓN)
        # Nota: Guardamos tal cual lo tenías, pero podrías querer guardar la predicción
        conn = get_connection()
        cursor = conn.cursor()
        movimientos_json = json.dumps(movimientos)

        sql = """
            INSERT INTO captcha_resultados (
                nombre, email, duracion_total_ms, longitud_trayectoria, 
                clics, label, tipo_fuente, movimientos_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Nota: en 'label' estamos guardando lo que venía del frontend (datos.get("label"))
        # Si quieres guardar lo que predijo el modelo, cambia datos.get("label") por es_bot
        valores = (
            datos.get("nombre"),
            datos.get("email"),
            datos.get("duracion_total_ms"),
            datos.get("longitud_trayectoria"),
            datos.get("clics"),
            datos.get("label"), 
            datos.get("tipo_fuente"),
            movimientos_json
        )
        cursor.execute(sql, valores)
        conn.commit()
        cursor.close()
        conn.close()

        # C) RESPUESTA CON REDIRECCIÓN (MODIFICADO)
        # Devolvemos la URL a la que el frontend debe ir
        return jsonify({
            "message": "Procesado",
            "redirect_url": f"/ver_resultado?nombre={datos.get('nombre')}&prediccion={etiqueta_predicha}&prob={probabilidad:.2f}"
        })

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Nueva ruta para mostrar el HTML del resultado
@app.route("/ver_resultado")
def ver_resultado():
    nombre = request.args.get('nombre', 'Usuario')
    prediccion = request.args.get('prediccion', 'DESCONOCIDO')
    prob = request.args.get('prob', '0')
    return render_template('resultado.html', nombre=nombre, prediccion=prediccion, prob=prob)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)