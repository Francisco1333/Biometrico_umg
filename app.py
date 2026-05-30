from flask import Flask, render_template, request, redirect, session, url_for
from registro_routes import init_registro_routes
from ingresos_routes import init_ingresos_routes
from bd_config import get_connection
from asistencias_routes import init_asistencia_routes

app = Flask(__name__)
app.secret_key = "umg_biometrico_2026"

init_registro_routes(app)
init_ingresos_routes(app)
init_asistencia_routes(app)
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        contrasena = request.form.get("contrasena", "").strip()
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute(
                "SELECT id_sede, Nombre FROM sedes WHERE username = %s AND contraseña = %s",
                (username, contrasena)
            )
            sede = cur.fetchone()
            cur.close(); conn.close()
            if sede:
                session['id_sede']     = sede[0]
                session['nombre_sede'] = sede[1]
                return redirect(url_for('index'))
            else:
                error = "Usuario o contrasena incorrectos"
        except Exception as e:
            print("Error login:", e)
            error = "Error de conexion"
    return render_template("login.html", error=error)

@app.route("/index")
def index():
    if 'id_sede' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", nombre_sede=session.get('nombre_sede'))

@app.route("/salir")
def salir():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    print("Servidor corriendo en: http://127.0.0.1:5000")
    app.run(debug=False)