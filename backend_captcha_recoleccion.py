
from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)
DATA_DIR = "datos_captcha"
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/captura", methods=["POST"])
def captura():
    datos = request.get_json()

    # Crear nombre de archivo por fecha
    fecha = datetime.now().strftime("%Y%m%d")
    nombre_archivo = os.path.join(DATA_DIR, f"capturas_{fecha}.csv")

    # Definir campos clave
    campos = ["nombre", "email", "duracion_total_ms", "longitud_trayectoria", "clics", "label", "tipo_fuente"]

    # Si el archivo no existe, se crea con cabecera
    archivo_nuevo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if archivo_nuevo:
            writer.writeheader()
        writer.writerow({k: datos.get(k, "") for k in campos})

    return jsonify({"mensaje": "Datos recibidos correctamente."})

@app.route("/", methods=["GET"])
def home():
    return "Backend activo para recolección de datos CAPTCHA"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
