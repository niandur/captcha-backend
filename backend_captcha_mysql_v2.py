import os
import json
import traceback
import logging, sys
logging.basicConfig(level=logging.INFO)

from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import mysql.connector

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

# ============================================================
# Config DB (RENDER: usa variables de entorno)
# ============================================================
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "qaot733.niandur.com"),
    "user": os.getenv("MYSQL_USER", "qaot733"),
    "password": os.getenv("MYSQL_PASSWORD", "Omega73bd"),
    "database": os.getenv("MYSQL_DATABASE", "qaot733"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}

TABLE_NAME = os.getenv("MYSQL_TABLE", "interacciones")

# ============================================================
# Modelo / Escalador
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_bot_final.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_final.pkl")

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)

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

# Si entrenaste con umbral distinto, cámbialo aquí
BOT_THRESHOLD = float(os.getenv("BOT_THRESHOLD", "0.5"))


# ============================================================
# Helpers: DB
# ============================================================
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def guardar_interaccion_mysql(payload: dict, es_bot: int, prob_bot: float):
    """
    Guarda la interacción completa en MySQL.
    Ajusta nombres de campos según tu tabla real.
    """

    try:
        conn = get_db_connection()
        print("DB_CONFIG:", {k: ("***" if "password" in k.lower() else v) for k,v in DB_CONFIG.items()})
        cur = conn.cursor()

        session_id = payload.get("session_id")
        user_agent = payload.get("user_agent") or request.headers.get("User-Agent", "")
        total_time = payload.get("total_time")
        clicks = payload.get("clicks")
        navigator_info = payload.get("navigator_info", {})

        mouse_data = payload.get("mouse_movements", [])

        # Guarda JSON como texto
        mouse_json = json.dumps(mouse_data, ensure_ascii=False)
        nav_json = json.dumps(navigator_info, ensure_ascii=False)

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logging.info("== MYSQL: intentando guardar ==")
        sys.stdout.flush()

        ok_db, err_db = guardar_interaccion_mysql(payload, es_bot, prob_bot)
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
                ts,                 # timestamp
                total_time,         # total_time
                mouse_json,         # mouse_data
                None,               # keystrokes_data (no lo usas → NULL)
                es_bot,
                float(prob_bot),
                nav_json,
            ),
        )
        conn.commit()
        logging.info(f"== MYSQL RESULT == ok={ok_db} err={err_db}")
        sys.stdout.flush()
        cur.close()
        conn.close()
        return True, None

    except Exception as e:
        return False, str(e)


# ============================================================
# Feature engineering (desde mouse_movements)
# ============================================================
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def calcular_features(payload: dict) -> dict:
    """
    Construye un dict con TODAS las FEATURES esperadas por el modelo.
    """
    points = payload.get("mouse_movements", [])
    num_clics = int(payload.get("clicks", 0))
    total_time_ms = _safe_float(payload.get("total_time", 0.0), 0.0)

    longitud_trayectoria = len(points)

    # Base de retorno (todo inicializado)
    feat = {c: 0.0 for c in FEATURES}
    feat["longitud_trayectoria"] = float(longitud_trayectoria)
    feat["tiempo_total"] = float(total_time_ms)

    # Si no hay suficientes puntos, devolvemos mínimos coherentes
    if longitud_trayectoria < 2:
        feat["ratio_clics"] = (num_clics / max(longitud_trayectoria, 1)) if total_time_ms >= 0 else 0.0
        # forma_estimativa no se puede inferir => se queda a 0
        return feat

    df = pd.DataFrame(points)

    # Asegurar columnas x,y,t
    for col in ("x", "y", "t"):
        if col not in df.columns:
            df[col] = 0.0

    df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["t"] = pd.to_numeric(df["t"], errors="coerce").fillna(0.0)

    # Orden por tiempo si aplica
    df = df.sort_values("t").reset_index(drop=True)

    dx = df["x"].diff()
    dy = df["y"].diff()
    dt = df["t"].diff()

    # Evitar dt=0
    dt = dt.replace(0, np.nan)

    # Distancia instantánea
    dist_inst = np.sqrt(dx**2 + dy**2).fillna(0.0)
    distancia_total = dist_inst.sum()

    # Si tu "distancia_total_norm" en entreno ya era normalizada de otro modo,
    # aquí deberías replicar EXACTAMENTE esa fórmula.
    # Como no tengo tu fórmula exacta, lo dejo como distancia_total "cruda"
    # y confío en el SCALER del modelo (es lo más habitual).
    feat["distancia_total_norm"] = float(distancia_total)

    # Velocidad instantánea (px por unidad t)
    vel_inst = (dist_inst / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feat["velocidad_media_norm"] = float(vel_inst.mean())
    feat["velocidad_std"] = float(vel_inst.std(ddof=0) if len(vel_inst) > 1 else 0.0)

    # Aceleración instantánea
    acc_inst = vel_inst.diff() / dt
    acc_inst = acc_inst.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["aceleracion_std"] = float(acc_inst.std(ddof=0) if len(acc_inst) > 1 else 0.0)

    # Ratio de clics: en tu notebook podía ser por puntos, por tiempo, etc.
    # Aquí lo dejo por número de puntos (estable) — si lo entrenaste distinto, ajusta.
    feat["ratio_clics"] = float(num_clics / max(longitud_trayectoria, 1))

    # Curvatura: cambios de ángulo entre segmentos
    ang = np.arctan2(dy.fillna(0.0), dx.fillna(0.0))
    d_ang = np.diff(ang)
    # normalizar a [-pi, pi]
    d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi
    d_ang_abs = np.abs(d_ang)

    feat["curvatura_media"] = float(np.mean(d_ang_abs)) if len(d_ang_abs) else 0.0
    feat["curvatura_std"] = float(np.std(d_ang_abs)) if len(d_ang_abs) else 0.0

    # Forma estimativa: ejemplo simple (ajusta a tu lógica del notebook si era distinta)
    # - lineal: baja curvatura media
    # - suave: curvatura moderada y baja std
    forma_lineal = 1.0 if feat["curvatura_media"] < 0.25 else 0.0
    forma_suave = 1.0 if (0.25 <= feat["curvatura_media"] < 0.60 and feat["curvatura_std"] < 0.50) else 0.0

    feat["forma_estimativa_lineal"] = forma_lineal
    feat["forma_estimativa_suave"] = forma_suave

    return feat


def build_df_final(features_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([features_dict])

    # Garantizar columnas y orden
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df = df[FEATURES]

    # Forzar numérico y limpiar NaNs
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ============================================================
# Routes
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    logging.info("== PREDICT: recibido JSON ==")
    logging.info(request.get_json(silent=True))
    sys.stdout.flush()
    try:
        payload = request.get_json(force=True) or {}
        if request.method == "OPTIONS":
            response = jsonify({"ok": True})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
            return response, 200
        # Fallback session_id si no viene
        if not payload.get("session_id"):
            payload["session_id"] = f"session_{int(datetime.utcnow().timestamp())}"

        # Fallback user_agent si no viene
        if not payload.get("user_agent"):
            payload["user_agent"] = request.headers.get("User-Agent", "")

        features_dict = calcular_features(payload)
        df_final = build_df_final(features_dict)

        # DEBUG temporal (quítalo cuando funcione)
        print("DF_FINAL_COLS:", df_final.columns.tolist())
        print("DF_FINAL_ROW:", df_final.iloc[0].to_dict())
        print("DF_FINAL_NAN:", df_final.isna().sum().to_dict())

        X_scaled = SCALER.transform(df_final.values)

        # Predicción
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X_scaled)[0]
            # Asumimos clase 1 = bot (ajusta si tu modelo codifica al revés)
            prob_bot = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # fallback sin probabilidades
            prob_bot = float(MODEL.predict(X_scaled)[0])

        es_bot = 1 if prob_bot >= BOT_THRESHOLD else 0
        es_humano = (es_bot == 0)

        # Guardar en MySQL
        ok_db, err_db = guardar_interaccion_mysql(payload, es_bot, prob_bot)
        if not ok_db:
            # No rompemos la predicción si la BD falla, pero lo devolvemos
            print("ERROR_MYSQL:", err_db)

        return jsonify(
            {
                "success": True,
                "is_human": es_humano,
                "bot_probability": prob_bot,
                "saved_to_db": ok_db,
                "db_error": err_db if not ok_db else None,
            }
        )

    except Exception as e:
        print("ERROR_PREDICT:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/resultado", methods=["GET"])
def resultado():
    """
    Renderiza resultado.html a partir de query params:
      /resultado?is_human=true&bot_probability=0.12
    """
    is_human = request.args.get("is_human", "false").lower() == "true"
    bot_probability = request.args.get("bot_probability", "0.0")
    return render_template(
        "resultado.html",
        is_human=is_human,
        bot_probability=bot_probability,
    )


if __name__ == "__main__":
    print("🚀 Iniciando Backend Captcha V2 (corregido)...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
