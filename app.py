from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Cargamos el modelo y las columnas seleccionadas
modelo = joblib.load(BASE_DIR / "modelo_car.pkl")
features = joblib.load(BASE_DIR / "features.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None

    # Información breve del modelo para mostrar en la página
    info_modelo = {
        "modelo": "RandomForestRegressor",
        "metodo": "Selección de características por correlación",
        "objetivo": "price",
        "mae": "1317.73",
        "r2": "0.9546"
    }

    # Ranking corto de variables importantes
    ranking_variables = [
        ("enginesize", "Tamaño del motor"),
        ("curbweight", "Peso del vehículo"),
        ("horsepower", "Caballos de fuerza"),
        ("carwidth", "Ancho del automóvil"),
        ("highwaympg", "Rendimiento en carretera"),
        ("citympg", "Rendimiento en ciudad"),
        ("carlength", "Largo del automóvil"),
        ("drivewheel", "Tipo de tracción"),
        ("wheelbase", "Distancia entre ejes"),
        ("boreratio", "Relación del cilindro del motor")
    ]

    # Descripción breve de los campos que sí se piden en el formulario
    variables_formulario = [
        ("wheelbase", "Distancia entre ejes del automóvil."),
        ("carlength", "Largo total del automóvil."),
        ("carwidth", "Ancho del automóvil."),
        ("curbweight", "Peso del automóvil sin carga adicional."),
        ("enginesize", "Tamaño del motor."),
        ("boreratio", "Relación del cilindro del motor."),
        ("horsepower", "Potencia del motor en caballos de fuerza."),
        ("citympg", "Rendimiento de combustible en ciudad."),
        ("highwaympg", "Rendimiento de combustible en carretera."),
        ("drivewheel", "Tipo de tracción del automóvil."),
        ("enginelocation", "Ubicación del motor."),
        ("enginetype", "Tipo de motor."),
        ("cylindernumber", "Número de cilindros."),
        ("fuelsystem", "Sistema de combustible."),
        ("carbody", "Tipo de carrocería.")
    ]

    def convertir_float(valor):
        """
        Convierte valores numéricos aceptando punto o coma decimal.
        Ejemplo: '88.6' o '88,6'
        """
        return float(valor.replace(",", "."))

    def convertir_int(valor):
        """
        Convierte valores numéricos a entero.
        """
        return int(float(valor.replace(",", ".")))

    if request.method == "POST":
        try:
            # Datos recibidos desde el formulario
            datos = {
                "wheelbase": convertir_float(request.form["wheelbase"]),
                "carlength": convertir_float(request.form["carlength"]),
                "carwidth": convertir_float(request.form["carwidth"]),
                "curbweight": convertir_int(request.form["curbweight"]),
                "enginesize": convertir_int(request.form["enginesize"]),
                "boreratio": convertir_float(request.form["boreratio"]),
                "horsepower": convertir_int(request.form["horsepower"]),
                "citympg": convertir_int(request.form["citympg"]),
                "highwaympg": convertir_int(request.form["highwaympg"]),
                "drivewheel": request.form["drivewheel"],
                "enginelocation": request.form["enginelocation"],
                "enginetype": request.form["enginetype"],
                "cylindernumber": request.form["cylindernumber"],
                "fuelsystem": request.form["fuelsystem"],
                "carbody": request.form["carbody"]
            }

            # Convertimos los datos a DataFrame
            entrada = pd.DataFrame([datos])

            # Convertimos variables categóricas a variables numéricas
            # IMPORTANTE: aquí se usa drop_first=False para que se genere la categoría seleccionada
            entrada = pd.get_dummies(entrada, drop_first=False)

            # Ajustamos las columnas para que coincidan con las usadas por el modelo
            entrada = entrada.reindex(columns=features, fill_value=0)

            # Realizamos predicción
            precio = modelo.predict(entrada)[0]

            # Redondeamos resultado
            prediccion = round(precio, 2)

        except Exception as e:
            error = f"Ocurrió un error al realizar la predicción: {e}"

    return render_template(
        "index.html",
        prediccion=prediccion,
        error=error,
        info_modelo=info_modelo,
        ranking_variables=ranking_variables,
        variables_formulario=variables_formulario
    )


if __name__ == "__main__":
    app.run(debug=True)