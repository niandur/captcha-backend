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
# Modelo / Scalers
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_rf_final.pkl")

# StandardScaler del notebook de entrenamiento (si lo usaste)
STD_SCALER_PATH = os.getenv("SCALER_PATH", "scaler_final.pkl")

# NUEVOS: artefactos que has generado del notebook de normalización
MINMAX_PATH = os.getenv("MINMAX_PATH", "minmax_scaler.pkl")
CLIP_BOUNDS_PATH = os.getenv("CLIP_BOUNDS_PATH", "clip_bounds.json")

BOT_THRESHOLD = float(os.getenv("BOT_THRESHOLD", "0.5"))

MODEL = joblib.load(MODEL_PATH)
STD_SCALER = joblib.load(STD_SCALER_PATH) if os.path.exists(STD_SCALER_PATH) else None
MINMAX_SCALER = joblib.load(MINMAX_PATH) if os.path.exists(MINMAX_PATH) else None

CLIP_BOUNDS = None
if os.path.exists(CLIP_BOUNDS_PATH):
    with open(CLIP_BOUNDS_PATH, "r", encoding="utf-8") as f:
        CLIP_BOUNDS = json.load(f)

logging.info(f"MODEL_PATH={MODEL_PATH} exists={os.path.exists(MODEL_PATH)}")
logging.info(f"STD_SCALER_PATH={STD_SCALER_PATH} exists={os.path.exists(STD_SCALER_PATH)}")
logging.info(f"MINMAX_PATH={MINMAX_PATH} exists={os.path.exists(MINMAX_PATH)}")
logging.info(f"CLIP_BOUNDS_PATH={CLIP_BOUNDS_PATH} exists={os.path.exists(CLIP_BOUNDS_PATH)}")
logging.info(f"MODEL.classes_={getattr(MODEL,'classes_',None)}")

# ============================================================
# Features EXACTAS que tu modelo espera
# (las que indicaste como definitivas)
# ============================================================
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
        total_time = payload.get("total_time")  # ms según frontend
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
                keystrokes_data,
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
# Feature engineering (RAW)
# ============================================================
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def calcular_features(payload: dict) -> dict:
    """
    Construye dict con TODAS las FEATURES esperadas por el modelo.
    El frontend debe enviar:
      - mouse_movements: lista {x,y,t} con t relativo (ms)
      - clicks: int
      - total_time: ms (duración)
    """
    points = payload.get("mouse_movements", [])
    clicks = int(payload.get("clicks", 0))
    total_time_ms = _safe_float(payload.get("total_time", 0.0), 0.0)

    feat = {c: 0.0 for c in FEATURES}

    longitud = len(points)
    feat["longitud_trayectoria"] = float(longitud)
    feat["tiempo_total"] = float(total_time_ms)

    # IMPORTANTE: este ratio debe coincidir con tu TFM.
    # Si en el notebook era clicks por tiempo: usa esto:
    # feat["ratio_clics"] = clicks / max(total_time_ms, 1.0)
    # Si era clicks por puntos (lo que tenías antes): usa esto:
    feat["ratio_clics"] = float(clicks / max(longitud, 1))

    if longitud < 2:
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
    distancia_total = float(dist_inst.sum())

    # OJO: aunque el nombre tenga _norm, aquí son RAW.
    feat["distancia_total_norm"] = distancia_total / max(longitud, 1)

    vel_inst = (dist_inst / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["velocidad_media_norm"] = float(vel_inst.mean())
    feat["velocidad_std"] = float(vel_inst.std(ddof=0) if len(vel_inst) > 1 else 0.0)

    acc_inst = (vel_inst.diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["aceleracion_std"] = float(acc_inst.std(ddof=0) if len(acc_inst) > 1 else 0.0)

    # Curvatura (cambios absolutos de ángulo)
    ang = np.arctan2(dy.fillna(0.0), dx.fillna(0.0))
    d_ang = np.diff(ang)
    d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi
    d_ang_abs = np.abs(d_ang)

    feat["curvatura_media"] = float(np.mean(d_ang_abs)) if len(d_ang_abs) else 0.0
    feat["curvatura_std"] = float(np.std(d_ang_abs)) if len(d_ang_abs) else 0.0

    feat["forma_estimativa_lineal"] = 0.0
    feat["forma_estimativa_suave"] = 0.0

    return feat


def build_df_final(features_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([features_dict])

    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    df = df[FEATURES]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ============================================================
# Normalización: clip + minmax + standard
# ============================================================
def clip_features(df: pd.DataFrame) -> pd.DataFrame:
# En producción No se aplica Clipping. Sólo en offline para estabilizar entrenamiento  
#  if not CLIP_BOUNDS:
#        return df
#
#    df2 = df.copy()
#    for col in FEATURES:
#        if col in CLIP_BOUNDS:
#            p01 = CLIP_BOUNDS[col].get("p01", None)
#            p99 = CLIP_BOUNDS[col].get("p99", None)
#            if p01 is not None and p99 is not None:
#                df2[col] = df2[col].clip(lower=float(p01), upper=float(p99))
    return df

def transform_pipeline(df_raw: pd.DataFrame):
    """
    Replica el pipeline del notebook:
      RAW -> clip -> MinMax.transform -> StandardScaler.transform -> X_scaled
    """
    df_clip = clip_features(df_raw)

    if MINMAX_SCALER is None:
        raise RuntimeError("MINMAX_SCALER no cargado. Revisa MINMAX_PATH.")

    X_minmax = MINMAX_SCALER.transform(df_clip[FEATURES])

    # Esta es la línea que decías que no encontrabas:
    # (StandardScaler aplicado DESPUÉS de MinMax, si así lo entrenaste)
    if STD_SCALER is not None:
        X_scaled = STD_SCALER.transform(X_minmax)
    else:
        X_scaled = X_minmax

    return df_clip, X_minmax, X_scaled


# ============================================================
# Routes
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "model_loaded": MODEL is not None,
            "std_scaler_loaded": STD_SCALER is not None,
            "minmax_loaded": MINMAX_SCALER is not None,
            "clip_bounds_loaded": CLIP_BOUNDS is not None,
            "model_classes": list(getattr(MODEL, "classes_", [])),
            "features": FEATURES,
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Preflight CORS
    if request.method == "OPTIONS":
        resp = jsonify({"ok": True})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return resp, 200

    try:
        payload = request.get_json(force=True) or {}

        if not payload.get("session_id"):
            payload["session_id"] = f"session_{int(datetime.utcnow().timestamp())}"

        if not payload.get("user_agent"):
            payload["user_agent"] = request.headers.get("User-Agent", "")

        # 1) features RAW
        features_dict = calcular_features(payload)
        df_raw = build_df_final(features_dict)

        logging.info("== FEATURES RAW ==")
        logging.info(df_raw.iloc[0].to_dict())
        sys.stdout.flush()

        # 2) pipeline normalización igual notebook
        df_clip, X_minmax, X_scaled = transform_pipeline(df_raw)

        logging.info("== FEATURES AFTER CLIP ==")
        logging.info(df_clip.iloc[0].to_dict())
        sys.stdout.flush()

        logging.info("== FEATURES AFTER MINMAX (primeras 5) ==")
        logging.info(X_minmax[0][:5].tolist())
        sys.stdout.flush()

        # 3) predicción robusta (sin invertir clases)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
        pred_label = MODEL.predict(X_scaled_df)[0]
        is_human = (pred_label == 1)
        es_bot = 0 if is_human else 1

        prob_human = None
        prob_bot = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X_scaled_df)[0]
            classes = list(getattr(MODEL, "classes_", []))

            if 1 in classes:
                prob_human = float(proba[classes.index(1)])
            if 0 in classes:
                prob_bot = float(proba[classes.index(0)])

        # fallback si no hay prob_bot por cualquier motivo
        if prob_bot is None and prob_human is not None:
            prob_bot = float(1.0 - prob_human)
        if prob_human is None and prob_bot is not None:
            prob_human = float(1.0 - prob_bot)

        # Umbral (opcional) — yo recomiendo usar pred_label, pero lo dejo por si lo quieres:
        # es_bot = 1 if (prob_bot is not None and prob_bot >= BOT_THRESHOLD) else 0
        # is_human = (es_bot == 0)

        # 4) guardar en MySQL
        ok_db, err_db = guardar_interaccion_mysql(payload, es_bot, float(prob_bot if prob_bot is not None else 0.0))
        logging.info(f"== MYSQL RESULT == ok={ok_db} err={err_db}")
        sys.stdout.flush()
        # --- FORZAR TIPOS JSON COMPATIBLES ---
        is_human = bool(is_human)
        prob_bot = float(prob_bot)
        prob_human = float(prob_human) if prob_human is not None else None
        # ------------------------------------

        # 5) respuesta (mantengo 'prob' para tu frontend actual)
        
        return jsonify(
            {
                "success": True,
                "is_human": is_human,
                "prob": float(prob_bot if prob_bot is not None else 0.0),  # compatibilidad frontend
                "prob_bot": prob_bot,
                "prob_human": prob_human,
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
    is_human = request.args.get("is_human", "false").lower() == "true"
    prob = request.args.get("prob", "0.0")
    return render_template("resultado.html", is_human=is_human, prob=prob)


if __name__ == "__main__":
    logging.info("Backend iniciado")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
