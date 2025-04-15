from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
# Cargar el dataset y crear la variable de "Eficiencia Global"
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# Cargar el dataset y crear la variable de "Eficiencia Global"
df = pd.read_csv('/Users/erickvanscoit/Github/Flask/data/F1_2022_data.csv')
import os

# Obtén la ruta del directorio donde se encuentra f122.py
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construye la ruta completa al archivo CSV en el directorio "data"
csv_path = os.path.join(base_dir, 'data', 'data/F1_2022_data.csv')

# Ahora lee el CSV usando la ruta relativa
df = pd.read_csv(csv_path)


# Definir la fórmula de Eficiencia Global:
# Eficiencia = (Wins + Podiums + 0.5 * No of Fastest Laps + 0.3 * Pole Positions) - (2 * DNFs)
df['Eficiencia'] = (
    df['Wins'] +
    df['Podiums'] +
    0.5 * df['No of Fastest Laps'] +
    0.3 * df['Pole Positions'] -
    2 * df['DNFs']
)

# Definir la variable predictora y el target
features = ['Eficiencia']
X = df[features]
y = df['Points']

# Crear un pipeline simple con escalado y regresión lineal
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Entrenar el modelo con todo el dataset (ten en cuenta que el dataset es pequeño)
pipeline_lr.fit(X, y)

# Obtener las predicciones y ordenar el ranking de pilotos
df['Predicted Points'] = pipeline_lr.predict(X)
df_ranking = df.sort_values(by='Predicted Points', ascending=False)

@app.route('/')
def index():
    # Convertir a tabla HTML el ranking que deseamos mostrar.
    # Mostramos solo el nombre del piloto, sus puntos reales y los predichos.
    ranking_table = df_ranking[['Driver Name', 'Points', 'Predicted Points']].to_html(classes='table table-striped', index=False)
    return render_template('index.html', ranking_table=ranking_table)

if __name__ == '__main__':
    app.run(debug=True)


