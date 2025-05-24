import sqlite3
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('spanish')

# ----------------------------
# 1. Crear la base de datos
# ----------------------------
conn = sqlite3.connect('municipalidad.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS tramites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ciudadano TEXT,
    descripcion TEXT,
    clasificacion TEXT,
    estado TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS cvs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    experiencia TEXT,
    puntaje REAL
)
''')

conn.commit()

# ----------------------------
# 2. Entrenar clasificador de trámites
# ----------------------------
datos = pd.DataFrame({
    'descripcion': [
        'Solicitud de licencia de construcción',
        'Pedido de limpieza de parque',
        'Permiso para eventos públicos',
        'Reclamo de luz pública',
        'Certificado de funcionamiento de negocio'
    ],
    'categoria': [
        'Licencias', 'Servicios Públicos', 'Eventos', 'Servicios Públicos', 'Licencias'
    ]
})

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(datos['descripcion'])
y = datos['categoria']

modelo = MultinomialNB()
modelo.fit(X, y)

# ----------------------------
# 3. Registrar trámite
# ----------------------------
def registrar_tramite(ciudadano, descripcion):
    vector = vectorizer.transform([descripcion])
    clasificacion = modelo.predict(vector)[0]
    estado = 'Recibido'

    cursor.execute("INSERT INTO tramites (ciudadano, descripcion, clasificacion, estado) VALUES (?, ?, ?, ?)",
                   (ciudadano, descripcion, clasificacion, estado))
    conn.commit()

    print(f"[NOTIFICACIÓN] {ciudadano}, su trámite ha sido clasificado como '{clasificacion}' y está en estado '{estado}'.")

# ----------------------------
# 4. Selección de CVs
# ----------------------------
def evaluar_cv(nombre, experiencia):
    # Simulamos el puntaje con NLP básico
    vector = vectorizer.transform([experiencia])
    puntaje = vector.sum()
    cursor.execute("INSERT INTO cvs (nombre, experiencia, puntaje) VALUES (?, ?, ?)", (nombre, experiencia, puntaje))
    conn.commit()
    print(f"[CV] {nombre} registrado con puntaje: {puntaje:.2f}")

def mostrar_top_candidatos(n=3):
    print("\nTOP CANDIDATOS:")
    for row in cursor.execute("SELECT nombre, puntaje FROM cvs ORDER BY puntaje DESC LIMIT ?", (n,)):
        print(f"{row[0]} - Puntaje: {row[1]:.2f}")

# ----------------------------
# 5. Simulación
# ----------------------------
registrar_tramite("Ana Torres", "Solicito certificado de funcionamiento")
registrar_tramite("Luis Rojas", "Falta de alumbrado público en mi calle")

evaluar_cv("Carlos Pérez", "Experiencia en gestión pública y desarrollo de software")
evaluar_cv("Lucía Vargas", "Contadora con experiencia en sector privado")
mostrar_top_candidatos()

# Cerrar conexión
conn.close()
