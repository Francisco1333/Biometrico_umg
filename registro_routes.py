from flask import request, jsonify, render_template
from bd_config import get_connection
from contextlib import contextmanager
import cv2
import os
import datetime
import numpy as np
from fpdf import FPDF
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import qrcode
import cloudinary
import cloudinary.uploader
import requests as req_http
import json
import tempfile

cloudinary.config(
    cloud_name="dspxagazw",
    api_key="332336868417792",
    api_secret="7M36RjNCZzoOZvm4w0PEUVHk0IM"
)

SENDGRID_API_KEY = "3Y7JL2QWL6QJ3VCRVGUUL2A9"
EMAIL_REMITENTE  = "francis14322@gmail.com"
CARNET_PLANTILLA_URL = "https://res.cloudinary.com/dspxagazw/image/upload/v1779922335/carnet_xfmjlg.png"

# ── Parámetros de comparación facial ──────────────────────────────────────
# Similitud coseno entre vectores HOG normalizados.
# 1.0 = imagen idéntica | 0.0 = completamente distinta
# Personas distintas: ~0.90-0.96
# Misma persona, iluminación diferente: ~0.97-0.999
# Umbral 0.97: solo marca duplicado si es prácticamente la misma persona
SIMILITUD_UMBRAL = 0.93

# HOG descriptor — 64×64 imagen, ventana 16×16, paso 8×8, 9 bins
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────

def generar_carnet_visual(persona_id):
    anio = datetime.datetime.now().year % 100
    return f"7691-{anio}-{persona_id:05d}"


def generar_qr(codigo):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    qrcode.make(codigo).save(tmp.name)
    tmp.close()
    return tmp.name


def _preprocesar(img_gris):
    """Redimensiona a 64×64 y aplica CLAHE para normalizar iluminación."""
    img = cv2.resize(img_gris, (64, 64))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def generar_encoding(img_gris):
    """
    Genera vector HOG L2-normalizado de 1764 dimensiones.
    Devuelve lista de floats, o None si la imagen no tiene textura (fondo liso).
    """
    img = _preprocesar(img_gris)
    h = _HOG.compute(img).flatten()
    norm = float(np.linalg.norm(h))
    if norm < 0.01:          # imagen sin textura = fondo, negro, etc.
        return None
    return (h / norm).tolist()


def detectar_cara(img_gris):
    """
    Intenta detectar la cara frontal más grande.
    Devuelve la región recortada en escala de grises, o None.
    NO usa la imagen completa como fallback — eso era la raíz del bug.
    """
    img_eq   = cv2.equalizeHist(img_gris)
    cascade  = cv2.CascadeClassifier(
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
    cara = img_gris[max(y+my,0):min(y+hf-my, h_img),
                    max(x+mx,0):min(x+wf-mx, w_img)]
    return cara if cara.size > 0 else None


def similitud_coseno(v1, v2):
    """
    Similitud coseno entre dos vectores ya normalizados (norma=1).
    Valor entre 0 y 1. Más alto = más parecidos.
    """
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    return float(np.dot(a, b))


def rostro_duplicado(cur, cara):
    """
    Compara el encoding HOG de 'cara' contra los registros en BD.
    Devuelve (True, str(id)) si hay duplicado, (False, None) si no.
    """
    if cara is None:
        print("[FACIAL] cara=None, sin comparación")
        return False, None

    encoding_nuevo = generar_encoding(cara)
    if encoding_nuevo is None:
        print("[FACIAL] encoding None (imagen sin textura), descartando")
        return False, None

    cur.execute(
        "SELECT id, encoding_facial FROM personas WHERE encoding_facial IS NOT NULL"
    )
    filas = cur.fetchall()
    if not filas:
        return False, None

    mejor_sim = -1.0
    mejor_id  = None

    for persona_id, encoding_json in filas:
        try:
            enc_bd = json.loads(encoding_json)
        except Exception:
            continue
        if enc_bd is None:
            continue

        sim = similitud_coseno(encoding_nuevo, enc_bd)
        print(f"[FACIAL] vs persona {persona_id} -> similitud={sim:.4f} (umbral={SIMILITUD_UMBRAL})")

        if sim > mejor_sim:
            mejor_sim = sim
            mejor_id  = str(persona_id)

    if mejor_sim >= SIMILITUD_UMBRAL:
        print(f"[FACIAL] *** DUPLICADO persona {mejor_id}, sim={mejor_sim:.4f} ***")
        return True, mejor_id

    print(f"[FACIAL] Sin duplicado. Mejor sim={mejor_sim:.4f}")
    return False, None


# ── PDF y correo ─────────────────────────────────────────────────────────────

def crear_pdf(datos):
    carnet_temp = qr_path = pdf_path = None
    try:
        resp = req_http.get(CARNET_PLANTILLA_URL, timeout=15)
        tmp_c = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_c.write(resp.content); tmp_c.close()
        carnet_temp = tmp_c.name

        carnet_visual = datos["carnet_visual"]
        qr_path = generar_qr(carnet_visual)

        tmp_p = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_p.close(); pdf_path = tmp_p.name

        pdf = FPDF(orientation='L', unit='mm', format='Letter')
        pdf.add_page()
        x, y, w, h = 30, 35, 220, 140
        pdf.image(carnet_temp, x=x, y=y, w=w, h=h)
        if os.path.exists(datos["foto_path"]):
            pdf.image(datos["foto_path"], x=x+12, y=y+28, w=45, h=55)
        pdf.image(qr_path, x=x+165, y=y+72, w=30, h=30)
        pdf.set_font("Arial", "B", 14)
        pdf.text(x+65, y+40, datos["nombre"])
        pdf.text(x+65, y+50, datos["apellido"])
        pdf.text(x+65, y+60, f"Carnet: {carnet_visual}")
        pdf.text(x+65, y+70, datos["carreras"])
        pdf.text(x+65, y+80, f"Seccion: {datos['seccion']}")
        pdf.output(pdf_path)
        return pdf_path
    finally:
        for tmp in [carnet_temp, qr_path]:
            if tmp and os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass


def enviar_correo_pdf(destino, pdf_path, carnet_visual):
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode()

        message = Mail(
            from_email=EMAIL_REMITENTE,
            to_emails=destino,
            subject="Su Carnet Estudiantil UMG",
            plain_text_content=(
                f"Estimado estudiante,\n\n"
                f"Adjunto encontrará su carnet estudiantil.\n"
                f"Número de carnet: {carnet_visual}\n\n"
                f"Universidad Mariano Gálvez de Guatemala."
            )
        )
        adjunto = Attachment(
            FileContent(pdf_data),
            FileName(f"carnet_{carnet_visual}.pdf"),
            FileType("application/pdf"),
            Disposition("attachment")
        )
        message.attachment = adjunto
        SendGridAPIClient(SENDGRID_API_KEY).send(message)
    except Exception as e:
        print("Error correo SendGrid:", e)


# ── Rutas ────────────────────────────────────────────────────────────────────

def init_registro_routes(app):

    @app.route("/registro")
    def registro():
        return render_template("registro.html")

    @app.route("/api/tipos")
    def api_tipos():
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id_tipo, tipo FROM tipos ORDER BY tipo")
                return jsonify([{"id_tipo": f[0], "tipo": f[1]} for f in cur.fetchall()])
        except Exception as e:
            print("Error api_tipos:", e)
            return jsonify([])

    @app.route("/api/carreras")
    def api_carreras():
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id_carrera, carrera FROM carreras ORDER BY carrera")
                return jsonify([{"id_carrera": f[0], "carrera": f[1]} for f in cur.fetchall()])
        except Exception as e:
            print("Error api_carreras:", e)
            return jsonify([])

    @app.route("/api/cursos/<int:id_carrera>")
    def api_cursos(id_carrera):
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id, nombre FROM cursos WHERE id_carrera = %s ORDER BY nombre",
                            (id_carrera,))
                return jsonify([{"id_curso": f[0], "nombre": f[1]} for f in cur.fetchall()])
        except Exception as e:
            print("Error api_cursos:", e)
            return jsonify([])

    # ── Diagnóstico: ver similitudes en tiempo real ─────────────────────────
    @app.route("/api/debug_facial", methods=["POST"])
    def api_debug_facial():
        """
        POST una foto y devuelve la similitud coseno contra todos los registros.
        Útil para calibrar SIMILITUD_UMBRAL.
        """
        try:
            archivo = request.files.get("foto")
            if not archivo:
                return jsonify({"error": "sin foto"})
            npimg    = np.frombuffer(archivo.read(), np.uint8)
            img_gris = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            cara     = detectar_cara(img_gris)
            enc      = generar_encoding(cara) if cara is not None else None

            if enc is None:
                return jsonify({"error": "cara no detectada o imagen sin textura"})

            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, nombre, apellido, encoding_facial "
                    "FROM personas WHERE encoding_facial IS NOT NULL"
                )
                filas = cur.fetchall()

            resultados = []
            for pid, nombre, apellido, enc_json in filas:
                try:
                    enc_bd = json.loads(enc_json)
                    if enc_bd is None: continue
                    sim = similitud_coseno(enc, enc_bd)
                    resultados.append({
                        "id": pid,
                        "nombre": f"{nombre} {apellido}",
                        "similitud": round(sim, 4),
                        "es_duplicado": sim >= SIMILITUD_UMBRAL
                    })
                except Exception:
                    pass

            resultados.sort(key=lambda r: r["similitud"], reverse=True)
            return jsonify({
                "umbral": SIMILITUD_UMBRAL,
                "cara_detectada": cara is not None,
                "resultados": resultados
            })
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/verificar_rostro", methods=["POST"])
    def api_verificar_rostro():
        try:
            archivo = request.files.get("foto")
            if not archivo:
                return jsonify({"duplicado": False})

            npimg    = np.frombuffer(archivo.read(), np.uint8)
            img_gris = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            if img_gris is None:
                return jsonify({"duplicado": False})

            cara = detectar_cara(img_gris)
            if cara is None:
                # No se detectó cara: NO marcar como duplicado, avisar al frontend
                return jsonify({"duplicado": False, "sin_cara": True})

            with db_conn() as conn:
                cur = conn.cursor()
                dup, pid = rostro_duplicado(cur, cara)
                return jsonify({"duplicado": dup, "persona_id": pid})

        except Exception as e:
            print("Error verificar_rostro:", e)
            return jsonify({"duplicado": False})

    @app.route("/api/registrar", methods=["POST"])
    def api_registrar():
        foto_temp_path = None
        pdf_path       = None
        try:
            nombre     = request.form["nombre"]
            apellido   = request.form["apellido"]
            telefono   = request.form["telefono"]
            email      = request.form["email"]
            seccion    = request.form["seccion"]
            id_tipo    = int(request.form["id_tipo"])
            tipo_texto = request.form["tipo_texto"].lower()

            ids_carreras     = [int(x) for x in request.form.getlist("carreras[]")]
            nombres_carreras = request.form.getlist("carreras_nombres[]")
            cursos           = list(set(int(x) for x in request.form.getlist("cursos[]")))

            archivo_foto = request.files.get("foto")
            foto_bytes   = archivo_foto.read()

            tmp_foto = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp_foto.write(foto_bytes); tmp_foto.close()
            foto_temp_path = tmp_foto.name

            img_gris = cv2.imdecode(np.frombuffer(foto_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img_gris is None:
                return jsonify({"ok": False, "error": "No se pudo procesar la foto."})

            cara          = detectar_cara(img_gris)
            enc           = generar_encoding(cara) if cara is not None else None
            encoding_json = json.dumps(enc)   # puede ser json.dumps(None) si no hay cara

            with db_conn() as conn:
                cur = conn.cursor()

                cur.execute(
                    "INSERT INTO personas "
                    "(nombre, apellido, telefono, email, foto_path, seccion, encoding_facial) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (nombre, apellido, telefono, email, "", seccion, encoding_json)
                )
                conn.commit()
                persona_id = cur.lastrowid

                cur.execute(
                    "INSERT INTO persona_tipo (id_persona, id_tipo) VALUES (%s,%s)",
                    (persona_id, id_tipo)
                )
                conn.commit()

                if tipo_texto == "estudiante":
                    for id_carrera in ids_carreras:
                        cur.execute(
                            "INSERT INTO persona_carrera (id_persona, id_carrera) "
                            "VALUES (%s,%s)", (persona_id, id_carrera)
                        )
                    for id_curso in cursos:
                        cur.execute(
                            "INSERT INTO estudiante_curso (id_persona, id_curso) "
                            "VALUES (%s,%s)", (persona_id, id_curso)
                        )
                    conn.commit()

                    carnet_visual  = generar_carnet_visual(persona_id)
                    texto_carreras = ", ".join(nombres_carreras) if nombres_carreras else "-"

                    pdf_path = crear_pdf({
                        "nombre": nombre, "apellido": apellido,
                        "foto_path": foto_temp_path, "seccion": seccion,
                        "carnet_visual": carnet_visual, "carreras": texto_carreras,
                    })
                    enviar_correo_pdf(email, pdf_path, carnet_visual)

                    with open(pdf_path, "rb") as f:
                        resultado = cloudinary.uploader.upload(
                            f, resource_type="raw", folder="umg_carnets",
                            public_id=f"carnet_{carnet_visual}", overwrite=True
                        )

                    cur.execute(
                        "UPDATE personas SET foto_path = %s WHERE id = %s",
                        (resultado["secure_url"], persona_id)
                    )
                    conn.commit()

                    return jsonify({
                        "ok": True,
                        "persona_id": persona_id,
                        "carnet_visual": carnet_visual
                    })

                elif tipo_texto == "catedratico":
                    for id_carrera in ids_carreras:
                        cur.execute(
                            "INSERT IGNORE INTO persona_carrera (id_persona, id_carrera) "
                            "VALUES (%s,%s)", (persona_id, id_carrera)
                        )
                    for id_curso in cursos:
                        cur.execute(
                            "INSERT IGNORE INTO catedratico_cursoaca (id_persona, id_curso) "
                            "VALUES (%s,%s)", (persona_id, id_curso)
                        )
                    conn.commit()

                return jsonify({"ok": True, "persona_id": persona_id})

        except Exception as e:
            print("Error api_registrar:", e)
            return jsonify({"ok": False, "error": str(e)})

        finally:
            for tmp in [foto_temp_path, pdf_path]:
                if tmp and os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass