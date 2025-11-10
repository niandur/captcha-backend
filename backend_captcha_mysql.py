
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import json
import traceback  # Para ver el error completo

app = Flask(__name__)
CORS(app, origins=["https://www.niandur.com", "http://127.0.0.1:5500"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
#CORS(app, origins=["https://www.niandur.com"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

@app.route("/captura", methods=["POST", "OPTIONS"])
def captura():
    if request.method == "OPTIONS":
        return '', 204  # Preflight CORS OK

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
        movimientos_json = json.dumps(datos.get("movimientos") or [])
        #movimientos_json = json.dumps(datos.get("movimientos", []))
        sql = """
            INSERT INTO captcha_resultados (
                nombre, email, duracion_total_ms,
                longitud_trayectoria, clics,
                label, tipo_fuente, fecha
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
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
        print("📥 Insertando valores:", valores)  # debug
        cursor.execute(sql, valores)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Datos insertados en la base de datos correctamente"})

    except Exception as e:
        print("❌ Error en backend:", e)
        traceback.print_exc()  # muestra dónde falla exactamente
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
