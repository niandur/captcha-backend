from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import json
import traceback  # Para ver el error completo

app = Flask(__name__)

# CORS abierto para simplificar (puedes restringir más adelante)
CORS(app)

def get_connection():
    return mysql.connector.connect(
        host="qaot733.niandur.com",
        user="qaot733",
        password="Omega73bd!",
        database="qaot733"
    )

@app.route("/guardar_captcha", methods=["POST", "OPTIONS"])
def guardar_captcha():
    # Preflight CORS
    if request.method == "OPTIONS":
        return "", 204

    datos = request.get_json()
    print("🧾 Recibido:", datos)

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Movimientos en JSON (tal cual llegan del frontend)
        movimientos_json = json.dumps(datos.get("movimientos") or [])

        sql = """
            INSERT INTO captcha_resultados (
                nombre,
                email,
                duracion_total_ms,
                longitud_trayectoria,
                clics,
                label,
                tipo_fuente,
                movimientos_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

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

        print("📥 Insertando valores:", valores)
        cursor.execute(sql, valores)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Datos insertados en la base de datos correctamente"})

    except Exception as e:
        print("❌ Error en backend:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)