
import os
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from datetime import datetime

# ---------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------

app = Flask(__name__)
from flask_cors import CORS

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://www.niandur.com",
            "https://niandur.com"
            "http://127.0.0.1:5500"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Carpeta donde guardamos los JSON de movimientos
DATA_FOLDER = "data_sessions"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Conexión MySQL
def get_connection():
    return mysql.connector.connect(
        host="qaot733.niandur.com",
        user="qaot733",
        password="Omega73bd!",
        database="qaot733"
    )


# ---------------------------------------------
# ENDPOINT PRINCIPAL
# ---------------------------------------------

@app.route('/guardar_captcha', methods=['POST'])
def guardar_captcha():

    try:
        data = request.json

        # Datos básicos enviados desde el frontend
        nombre = data.get("nombre", "")
        respuesta_correcta = data.get("respuesta_correcta")
        respuesta_usuario = data.get("respuesta_usuario")
        movimientos = data.get("movimientos", [])
        tipo_fuente = data.get("tipo_fuente", "web_humano")
        screen_width = data.get("screen_width", None)
        screen_height = data.get("screen_height", None)

        # Campos calculados
        duracion_total = data.get("duracion_total", 0)
        longitud_trayectoria = data.get("longitud_trayectoria", 0)

        # Etiqueta automática: humano = 1, bot = 0
        label = 1 if tipo_fuente.startswith("humano") else 0

        # ID único para la sesión
        session_id = str(uuid.uuid4())

        # Ruta del archivo JSON donde guardamos los movimientos
        json_path = os.path.join(DATA_FOLDER, f"{session_id}.json")

        # Creamos el JSON con estructura estándar compatible con Web Bot Dataset
        json_data = {
            "session_id": session_id,
            "source": tipo_fuente,
            "events": movimientos,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Guardar archivo JSON
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # ---------------------------------------------
        # GUARDAR METADATOS EN MySQL
        # ---------------------------------------------
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            INSERT INTO captcha_resultados 
            (id_sesion, nombre, respuesta_correcta, respuesta_usuario,
             duracion_total, longitud_trayectoria, filepath_json,
             tipo_fuente, label, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """

        values = (
            session_id,
            nombre,
            respuesta_correcta,
            respuesta_usuario,
            duracion_total,
            longitud_trayectoria,
            json_path,
            tipo_fuente,
            label,
        )

        cursor.execute(query, values)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "json_path": json_path
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------
# MAIN
# ---------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
