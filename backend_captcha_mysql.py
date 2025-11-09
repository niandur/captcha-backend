
from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins=["http://www.niandur.com"])
# Configuración de la base de datos MySQL (puede usarse con variables de entorno)
DB_CONFIG = {
    "host": "qaot733.niandur.com",
    "user": "qaot733",
    "password": "Omega73bd!",
    "database": "qaot733"
}

@app.route("/captura", methods=["POST"])
def captura():
    datos = request.get_json()

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
        INSERT INTO captcha_resultados (
            nombre, email, duracion_total_ms,
            longitud_trayectoria, clics, label, tipo_fuente, fecha
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        valores = (
            datos.get("nombre", ""),
            datos.get("email", ""),
            float(datos.get("duracion_total_ms", 0)),
            int(datos.get("longitud_trayectoria", 0)),
            int(datos.get("clics", 0)),
            int(datos.get("label", 0)),
            datos.get("tipo_fuente", "web"),
            datetime.now()
        )

        cursor.execute(query, valores)
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"mensaje": "Datos insertados en la base de datos correctamente."})

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Backend activo y conectado a MySQL"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
