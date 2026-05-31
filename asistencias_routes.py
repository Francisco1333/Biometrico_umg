from flask import request, jsonify, render_template, session, redirect, url_for
from bd_config import get_connection
import cv2
import datetime
import numpy as np
import json
import traceback

COOLDOWN_MINUTOS = 2
_ultimo_asistencia = {}

SIMILITUD_UMBRAL = 0.93
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

RANGOS_HORARIO = [
    ("7-9",   datetime.time(7,  0), datetime.time(9,  0)),
    ("9-11",  datetime.time(9,  1), datetime.time(11, 0)),
    ("11-13", datetime.time(11, 1), datetime.time(13, 0)),
    ("14-16", datetime.time(14, 0), datetime.time(16, 0)),
    ("16-18", datetime.time(16, 1), datetime.time(18, 0)),
    ("20-22", datetime.time(20, 1), datetime.time(22, 0)),
]

def obtener_horario_actual():
    hora = datetime.datetime.now().time()
    for codigo, inicio, fin in RANGOS_HORARIO:
        if inicio <= hora <= fin:
            return codigo
    return None

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


def init_asistencia_routes(app):

    @app.route("/asistencias")
    def asistencia_pagina():
        if 'id_sede' not in session:
            return redirect(url_for('login'))
        return render_template("asistencias.html", id_sede=session.get('id_sede', 1))

    @app.route("/asistencias/reconocer", methods=["POST"])
    def asistencia_reconocer():
        conn = None
        try:
            id_sede = int(request.form.get("id_sede", 1))
            archivo = request.files.get("frame")
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

            # Solo estudiantes
            cur.execute("""
                SELECT p.id, p.nombre, p.apellido, p.encoding_facial
                FROM personas p
                JOIN persona_tipo pt ON pt.id_persona = p.id
                JOIN tipos t ON t.id_tipo = pt.id_tipo
                WHERE p.encoding_facial IS NOT NULL AND t.tipo = 'Estudiante'
            """)
            filas = cur.fetchall()

            if not filas:
                return jsonify({"estado": "desconocido"})

            mejor_sim    = -1.0
            mejor_id     = None
            mejor_nombre = None

            for persona_id, nombre, apellido, encoding_json in filas:
                try:
                    enc_bd = json.loads(encoding_json)
                    if enc_bd is None:
                        continue
                    sim = similitud_coseno(enc_nuevo, enc_bd)
                    print(f"[ASIST] {persona_id} {nombre} {apellido} -> sim={sim:.4f}")
                    if sim > mejor_sim:
                        mejor_sim    = sim
                        mejor_id     = persona_id
                        mejor_nombre = f"{nombre} {apellido}"
                except Exception:
                    continue

            print(f"[ASIST] Mejor -> {mejor_id}, sim={mejor_sim:.4f} (umbral={SIMILITUD_UMBRAL})")

            if mejor_sim < SIMILITUD_UMBRAL or mejor_id is None:
                return jsonify({"estado": "desconocido"})

            # Buscar cualquier curso del estudiante sin restricción de horario
            cur.execute("""
                SELECT c.id, c.nombre FROM cursos c
                JOIN estudiante_curso ec ON ec.id_curso = c.id
                WHERE ec.id_persona = %s
                LIMIT 1
            """, (mejor_id,))
            curso_row = cur.fetchone()

            if not curso_row:
                return jsonify({"estado": "sin_curso", "nombre": mejor_nombre})

            curso_id     = curso_row[0]
            nombre_curso = curso_row[1]

            # FIX: usar parámetro Python en vez de CURDATE()
            hoy = datetime.date.today()

            # Verificar asistencia hoy
            cur.execute("""
                SELECT id FROM asistencias_curso
                WHERE estudiante_id = %s AND curso_id = %s AND DATE(fecha) = %s
            """, (mejor_id, curso_id, hoy))
            if cur.fetchone():
                return jsonify({"estado": "ya_registrado",
                                "nombre": mejor_nombre, "curso": nombre_curso})

            # Cooldown local
            ahora  = datetime.datetime.now()
            ultimo = _ultimo_asistencia.get((mejor_id, curso_id))
            if ultimo:
                diff = (ahora - ultimo).total_seconds() / 60
                if diff < COOLDOWN_MINUTOS:
                    return jsonify({"estado": "cooldown", "nombre": mejor_nombre,
                                    "restante": round(COOLDOWN_MINUTOS - diff, 1)})

            # FIX: estado='presente' en vez de None, hoy como parámetro
            cur.execute("""
                INSERT INTO asistencias_curso
                    (estudiante_id, curso_id, fecha, estado, id_sede)
                VALUES (%s, %s, %s, %s, %s)
            """, (mejor_id, curso_id, hoy, None, id_sede))
            conn.commit()
            _ultimo_asistencia[(mejor_id, curso_id)] = ahora

            return jsonify({"estado": "ok", "nombre": mejor_nombre, "curso": nombre_curso})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"estado": "error", "detalle": str(e)})
        finally:
            if conn:
                try: conn.close()
                except: pass
