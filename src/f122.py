import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Configurar el título de la aplicación
st.title("Ranking de Pilotos de Fórmula 1 (F1) 2022")

# Cargar el dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'data', '..data/F1_2022_data.csv')

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"No se encontró el archivo en la ruta: {csv_path}")
    st.stop()

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

# Mostrar el ranking en Streamlit
st.subheader("Ranking de Pilotos")
st.dataframe(df_ranking[['Driver Name', 'Points', 'Predicted Points']])

# Agregar una opción de visualización interactiva
if st.checkbox("Mostrar datos completos"):
    st.write("Datos del dataset:")
    st.dataframe(df)

# Información adicional
st.sidebar.title("Opciones")
st.sidebar.write("Usa las opciones para interactuar con los datos.")
