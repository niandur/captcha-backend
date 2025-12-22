import os
import json
import traceback
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import mysql.connector

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO)

# ============================================================
# Flask
# ============================================================
app = Flask(__name__, template_folder="templates")

# CORS global (por si acaso; preflight lo atendemos también explícitamente)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================
# Config DB (Render variables de entorno)
# ============================================================
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", ""),
    "user": os.getenv("MYSQL_USER", ""),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", ""),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}

TABLE_NAME = os.getenv("MYSQL_TABLE", "interacciones")

# ============================================================
# Modelo / Escalador
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_final.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_final.pkl")

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)

# Features EXACTAS que tu modelo espera (según lo que indicaste)
FEATURES = [
    "longitud_trayectoria",
    "distancia_total_norm",
    "velocidad_media_norm",
    "velocidad_std",
    "aceleracion_std",
    "ratio_clics",
    "curvatura_media",
    "curvatura_std",
    "tiempo_total",
    "forma_estimativa_lineal",
    "forma_estimativa_suave",
]

BOT_THRESHOLD = float(os.getenv("BOT_THRESHOLD", "0.5"))  # ajustable


# ============================================================
# Helpers DB
# ============================================================
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def guardar_interaccion_mysql(payload: dict, es_bot: int, prob_bot: float):
    """
    Inserta en tu tabla actual, cuyas columnas son:
    id, session_id, user_agent, timestamp, total_time, mouse_data, keystrokes_data,
    es_bot_prediccion, probabilidad_bot, navegador_info
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        session_id = payload.get("session_id")
        user_agent = payload.get("user_agent") or request.headers.get("User-Agent", "")
        total_time = payload.get("total_time")  # ms (según frontend)
        clicks = payload.get("clicks", 0)
        navigator_info = payload.get("navigator_info", {})
        mouse_data = payload.get("mouse_movements", [])
        keystrokes_data = payload.get("keystrokes_data", None)

        mouse_json = json.dumps(mouse_data, ensure_ascii=False)
        nav_json = json.dumps(navigator_info, ensure_ascii=False)

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        sql = f"""
        INSERT INTO {TABLE_NAME}
        (
            session_id,
            user_agent,
            timestamp,
            total_time,
            mouse_data,
            keystrokes_data,
            es_bot_prediccion,
            probabilidad_bot,
            navegador_info
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        cur.execute(
            sql,
            (
                session_id,
                user_agent,
                ts,
                total_time,
                mouse_json,
                keystrokes_data,     # None si no lo usas
                int(es_bot),
                float(prob_bot),
                nav_json,
            ),
        )

        conn.commit()
        cur.close()
        conn.close()
        return True, None

    except Exception as e:
        return False, str(e)


# ============================================================
# Feature engineering
# ============================================================
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def calcular_features(payload: dict) -> dict:
    """
    Construye un dict con TODAS las FEATURES esperadas por el modelo.
    El frontend debe enviar:
      - mouse_movements: lista de {x,y,t}
      - clicks: int
      - total_time: ms
    """
    points = payload.get("mouse_movements", [])
    clicks = int(payload.get("clicks", 0))
    total_time_ms = _safe_float(payload.get("total_time", 0.0), 0.0)

    feat = {c: 0.0 for c in FEATURES}

    longitud = len(points)
    feat["longitud_trayectoria"] = float(longitud)
    feat["tiempo_total"] = float(total_time_ms)

    # Ratio clicks (mantenemos consistente con tu pipeline; si lo entrenaste por tiempo, ajústalo)
    feat["ratio_clics"] = float(clicks / max(longitud, 1))

    if longitud < 2:
        # No hay suficiente info para curvatura/velocidad; dejamos 0
        return feat

    df = pd.DataFrame(points)

    for col in ("x", "y", "t"):
        if col not in df.columns:
            df[col] = 0.0

    df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["t"] = pd.to_numeric(df["t"], errors="coerce").fillna(0.0)

    df = df.sort_values("t").reset_index(drop=True)

    dx = df["x"].diff()
    dy = df["y"].diff()
    dt = df["t"].diff().replace(0, np.nan)

    dist_inst = np.sqrt(dx**2 + dy**2).fillna(0.0)
    distancia_total = dist_inst.sum()

    # OJO: aquí ponemos distancia_total en "distancia_total_norm".
    # Si en tu entrenamiento la "norm" era otra fórmula, aquí debes replicarla.
    feat["distancia_total_norm"] = float(distancia_total)

    vel_inst = (dist_inst / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["velocidad_media_norm"] = float(vel_inst.mean())
    feat["velocidad_std"] = float(vel_inst.std(ddof=0) if len(vel_inst) > 1 else 0.0)

    acc_inst = (vel_inst.diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["aceleracion_std"] = float(acc_inst.std(ddof=0) if len(acc_inst) > 1 else 0.0)

    # Curvatura: cambios absolutos de ángulo
    ang = np.arctan2(dy.fillna(0.0), dx.fillna(0.0))
    d_ang = np.diff(ang)
    d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi
    d_ang_abs = np.abs(d_ang)

    feat["curvatura_media"] = float(np.mean(d_ang_abs)) if len(d_ang_abs) else 0.0
    feat["curvatura_std"] = float(np.std(d_ang_abs)) if len(d_ang_abs) else 0.0

    # One-hot de forma estimativa (mínimo viable)
    # Si en tu notebook la forma se calculaba distinto, aquí debes copiar exactamente esa lógica.
    # Esto al menos asegura que las columnas existen y son numéricas.
    forma_lineal = 1.0 if feat["curvatura_media"] < 0.25 else 0.0
    forma_suave = 1.0 if (0.25 <= feat["curvatura_media"] < 0.60 and feat["curvatura_std"] < 0.50) else 0.0
    feat["forma_estimativa_lineal"] = forma_lineal
    feat["forma_estimativa_suave"] = forma_suave

    return feat


def build_df_final(features_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([features_dict])

    # Asegurar todas las columnas y orden correcto
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    df = df[FEATURES]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ============================================================
# Routes
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Preflight
    if request.method == "OPTIONS":
        resp = jsonify({"ok": True})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return resp, 200

    try:
        payload = request.get_json(force=True) or {}

        # Fallback session_id
        if not payload.get("session_id"):
            payload["session_id"] = f"session_{int(datetime.utcnow().timestamp())}"

        # Fallback user_agent
        if not payload.get("user_agent"):
            payload["user_agent"] = request.headers.get("User-Agent", "")

        logging.info("== PREDICT: recibido JSON ==")
        logging.info(payload)
        sys.stdout.flush()
        logging.info(f"MYSQL_HOST={os.getenv('MYSQL_HOST')}")
        logging.info(f"MYSQL_DATABASE={os.getenv('MYSQL_DATABASE')}")
        logging.info(f"MYSQL_USER={os.getenv('MYSQL_USER')}")
        # Features
        features_dict = calcular_features(payload)
        df_final = build_df_final(features_dict)

        logging.info("== DF_FINAL (1 fila) ==")
        logging.info(df_final.iloc[0].to_dict())
        sys.stdout.flush()

        # Escalado (manteniendo nombres para evitar warnings)
        X_scaled = SCALER.transform(df_final[FEATURES])

    # =========================
    # Predicción (robusta)
    # =========================
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X_scaled)[0]
        classes = list(getattr(MODEL, "classes_", []))  # normalmente [0, 1]

        # En tu dataset: label 1 = HUMANO, label 0 = BOT
        prob_human = float(proba[classes.index(1)]) if 1 in classes else None
        prob_bot = float(proba[classes.index(0)]) if 0 in classes else None

        # Predicción final (más fiable que umbral si hay lío de clases)
        pred_label = int(MODEL.predict(X_scaled)[0])  # 1=humano, 0=bot
    else:
        # Sin predict_proba: usamos predict directamente
        pred_label = int(MODEL.predict(X_scaled)[0])
        prob_human = None
        prob_bot = None

    is_human = (pred_label == 1)
    es_bot = 0 if is_human else 1

    # =========================
    # Guardar en MySQL
    # =========================
    ok_db, err_db = guardar_interaccion_mysql(payload, es_bot, prob_bot if prob_bot is not None else float(1 - pred_label))
    logging.info(f"== MYSQL RESULT == ok={ok_db} err={err_db}")
    sys.stdout.flush()

    # =========================
    # Respuesta
    # =========================
    return jsonify(
        {
            "success": True,
            "is_human": is_human,
            "prob_human": prob_human,
            "prob_bot": prob_bot,
            "saved_to_db": ok_db,
            "db_error": err_db,
        }
    ), 200

    except Exception as e:
        logging.error("ERROR /predict: %s", str(e))
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/resultado", methods=["GET"])
def resultado():
    """
    Si llamas así:
      /resultado?is_human=true&prob=0.12
    renderiza resultado.html
    """
    is_human = request.args.get("is_human", "false").lower() == "true"
    prob = request.args.get("prob", "0.0")
    return render_template("resultado.html", is_human=is_human, prob=prob)


if __name__ == "__main__":
    logging.info("🚀 Backend iniciado")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
