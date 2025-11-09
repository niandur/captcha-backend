
from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins=["https://www.niandur.com"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
# Configuración de la base de datos MySQL (puede usarse con variables de entorno)
DB_CONFIG = {
    "host": "qaot733.niandur.com",
    "user": "qaot733",
    "password": "Omega73bd!",
    "database": "qaot733"
}

@app.route("/captura", methods=["POST", "OPTIONS"])
def captura():
    def captura():
    if request.method == "OPTIONS":
        return '', 204  # Respuesta vacía para preflight
    datos = request.get_json()
    print("🧾 Recibido:", datos)

       try:
        conn = mysql.connector.connect(
            host="qaot733.niandur.com",
            user="qaot733",
            password="Omega73bd!",
            database="qaot733"
        )
        cursor = conn.cursor()

        sql = """
            INSERT INTO captcha_resultados (
                nombre, email, duracion_total_ms,
                longitud_trayectoria, clics,
                label, tipo_fuente, fecha
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """

        valores = (
            datos.get("nombre"),
            datos.get("email"),
            datos.get("duracion_total_ms"),
            datos.get("longitud_trayectoria"),
            datos.get("clics"),
            datos.get("label"),
            datos.get("tipo_fuente")
        )

        cursor.execute(sql, valores)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"mensaje": "Datos insertados en la base de datos correctamente."})

    except Exception as e:
        print("❌ Error en backend:", e)
        return jsonify({"error": str(e)}), 500
@app.route("/", methods=["GET"])
def home():
    return "Backend activo y conectado a MySQL"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
