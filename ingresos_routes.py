from flask import request, jsonify, render_template, session, redirect, url_for
from bd_config import get_connection
import cv2
import datetime
import numpy as np
import json

COOLDOWN_MINUTOS = 2
_ultimo_registrado = {}

# ── Mismo HOG que registro_routes.py ─────────────────────────────────────
SIMILITUD_UMBRAL = 0.93
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

def _preprocesar(img_gris):
    img = cv2.resize(img_gris, (64, 64))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def generar_encoding(img_gris):
    img = _preprocesar(img_gris)
    h = _HOG.compute(img).flatten()
    norm = float(np.linalg.norm(h))
    if norm < 0.01:
        return None
    return (h / norm).tolist()

def detectar_cara(img_gris):
    img_eq  = cv2.equalizeHist(img_gris)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    h_img, w_img = img_gris.shape
    min_sz = max(30, min(h_img, w_img) // 5)

    faces = cascade.detectMultiScale(img_eq, scaleFactor=1.05,
                                     minNeighbors=4, minSize=(min_sz, min_sz))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(img_eq, scaleFactor=1.1,
                                         minNeighbors=3, minSize=(min_sz, min_sz))
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, wf, hf = faces[0]
    mx, my = int(wf * 0.08), int(hf * 0.08)
    cara = img_gris[max(y+my, 0):min(y+hf-my, h_img),
                    max(x+mx, 0):min(x+wf-mx, w_img)]
    return cara if cara.size > 0 else None

def similitud_coseno(v1, v2):
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    return float(np.dot(a, b))


def init_ingresos_routes(app):

    @app.route("/ingresos/<ubicacion>")
    def ingresos(ubicacion):
        if 'id_sede' not in session:
            return redirect(url_for('login'))
        return render_template("ingresos.html", ubicacion=ubicacion,
                               id_sede=session.get('id_sede', 1))

    @app.route("/api/reconocer", methods=["POST"])
    def api_reconocer():
        conn = None
        try:
            ubicacion = request.form.get("ubicacion", "puerta")
            id_sede   = int(request.form.get("id_sede", 1))
            archivo   = request.files.get("frame")
            if not archivo:
                return jsonify({"estado": "sin_frame"})

            npimg    = np.frombuffer(archivo.read(), np.uint8)
            img_gris = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            if img_gris is None:
                return jsonify({"estado": "sin_frame"})

            cara = detectar_cara(img_gris)
            if cara is None:
                return jsonify({"estado": "sin_rostro"})

            enc_nuevo = generar_encoding(cara)
            if enc_nuevo is None:
                return jsonify({"estado": "sin_rostro"})

            conn = get_connection()
            cur  = conn.cursor()

            cur.execute(
                "SELECT id, nombre, apellido, encoding_facial "
                "FROM personas WHERE encoding_facial IS NOT NULL"
            )
            filas = cur.fetchall()

            if not filas:
                return jsonify({"estado": "desconocido"})

            mejor_sim  = -1.0
            mejor_id   = None
            mejor_nombre = None

            for persona_id, nombre, apellido, encoding_json in filas:
                try:
                    enc_bd = json.loads(encoding_json)
                    if enc_bd is None:
                        continue
                    sim = similitud_coseno(enc_nuevo, enc_bd)
                    print(f"[INGRESO] {persona_id} {nombre} {apellido} -> sim={sim:.4f}")
                    if sim > mejor_sim:
                        mejor_sim    = sim
                        mejor_id     = persona_id
                        mejor_nombre = f"{nombre} {apellido}"
                except Exception:
                    continue

            print(f"[INGRESO] Mejor -> {mejor_id}, sim={mejor_sim:.4f} (umbral={SIMILITUD_UMBRAL})")

            if mejor_sim < SIMILITUD_UMBRAL or mejor_id is None:
                return jsonify({"estado": "desconocido"})

            ahora = datetime.datetime.now()

            # Cooldown local
            ultimo_local = _ultimo_registrado.get(mejor_id)
            if ultimo_local:
                diff_local = (ahora - ultimo_local).total_seconds() / 60
                if diff_local <= COOLDOWN_MINUTOS:
                    restante = COOLDOWN_MINUTOS - diff_local
                    return jsonify({"estado": "cooldown", "nombre": mejor_nombre,
                                    "restante": round(restante, 1)})

            # Cooldown BD
            cur.execute(
                "SELECT timestamp FROM registros_ingreso "
                "WHERE persona_id = %s AND ubicacion = %s "
                "ORDER BY timestamp DESC LIMIT 1",
                (mejor_id, ubicacion)
            )
            row = cur.fetchone()
            if row:
                diff = (ahora - row[0]).total_seconds() / 60
                if diff <= COOLDOWN_MINUTOS:
                    restante = COOLDOWN_MINUTOS - diff
                    _ultimo_registrado[mejor_id] = row[0]
                    return jsonify({"estado": "cooldown", "nombre": mejor_nombre,
                                    "restante": round(restante, 1)})

            # Registrar ingreso
            cur.execute(
                "INSERT INTO registros_ingreso "
                "(persona_id, ubicacion, timestamp, id_sede) VALUES (%s,%s,%s,%s)",
                (mejor_id, ubicacion, ahora, id_sede)
            )
            conn.commit()
            _ultimo_registrado[mejor_id] = ahora

            return jsonify({"estado": "ok", "nombre": mejor_nombre, "persona_id": mejor_id})

        except Exception as e:
            print("Error api_reconocer:", e)
            return jsonify({"estado": "error"})
        finally:
            if conn:
                try: conn.close()
                except: pass