import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Ruta del CSV dentro del proyecto
ruta_csv = BASE_DIR / "data" / "CarPrice_Assignment.csv"

print("Buscando dataset en:", ruta_csv)

# Cargamos el dataset
df = pd.read_csv(ruta_csv)

print("Dataset cargado correctamente")
print(df.head())

# Creamos copia
updated_df = df.copy()

# Eliminamos columnas que no se usarán
updated_df = updated_df.drop(columns=['car_ID', 'CarName'])

# Convertimos variables categóricas a numéricas
updated_df = pd.get_dummies(updated_df, drop_first=True)

# Convertimos solamente columnas booleanas a enteros
columnas_booleanas = updated_df.select_dtypes(include='bool').columns
updated_df[columnas_booleanas] = updated_df[columnas_booleanas].astype(int)

# Calculamos matriz de correlación
matriz_corr = updated_df.corr()

# Tomamos correlaciones con price
correlaciones_price = matriz_corr['price'].drop('price')

# Valor absoluto y ordenamiento
correlaciones_price_ordenadas = correlaciones_price.abs().sort_values(ascending=False)

# Seleccionamos las 20 variables más correlacionadas
features_seleccionadas = correlaciones_price_ordenadas.head(20).index.tolist()

# Creamos X e y
X = updated_df[features_seleccionadas]
y = updated_df['price']

# Creamos el modelo final
modelo = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# Entrenamos el modelo
modelo.fit(X, y)

# Guardamos modelo y columnas
joblib.dump(modelo, BASE_DIR / "modelo_car.pkl")
joblib.dump(features_seleccionadas, BASE_DIR / "features.pkl")

print("Modelo entrenado y guardado correctamente.")
print("Variables usadas:")
print(features_seleccionadas)