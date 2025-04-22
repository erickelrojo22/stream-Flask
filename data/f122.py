from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el dataset y construir la variable de "Eficiencia Global"
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'data', 'F1_2022_data.csv')  # Asegurarse de que el archivo esté correctamente ubicado

# Leer el archivo CSV
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise Exception(f"No se encontró el archivo en la ruta: {csv_path}")

# Calcular la "Eficiencia Global"
df['Eficiencia'] = (
    df['Wins'] +
    df['Podiums'] +
    0.5 * df['No of Fastest Laps'] +
    0.3 * df['Pole Positions'] -
    2 * df['DNFs']
)

# Definir las características y la variable objetivo
features = ['Eficiencia']
X = df[features]
y = df['Points']

# Crear y entrenar un pipeline de regresión lineal
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline_lr.fit(X, y)

# Predicción de puntos y clasificación de pilotos
df['Predicted Points'] = pipeline_lr.predict(X)
df_ranking = df.sort_values(by='Predicted Points', ascending=False)

# Ruta principal para mostrar el ranking en formato HTML
@app.route('/')
def index():
    ranking_table = df_ranking[['Driver Name', 'Points', 'Predicted Points']].to_html(classes='table table-striped', index=False)
    return render_template('index.html', ranking_table=ranking_table)

# Configurar el puerto y el host para Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usar el puerto especificado por Render
    app.run(host="0.0.0.0", port=port)
