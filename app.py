from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargamos el modelo y las columnas seleccionadas
modelo = joblib.load("modelo_car.pkl")
features = joblib.load("features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None

    # Información para mostrar en la página
    info_modelo = {
        "modelo": "RandomForestRegressor",
        "metodo": "Selección de características por correlación con la variable price",
        "objetivo": "Predecir el precio de un automóvil",
        "mae": "1317.73",
        "r2": "0.9546"
    }

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
        return float(valor.replace(",", "."))

    def convertir_int(valor):
        return int(float(valor.replace(",", ".")))

    if request.method == "POST":
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

        entrada = pd.DataFrame([datos])

        entrada = pd.get_dummies(entrada, drop_first=True)

        entrada = entrada.reindex(columns=features, fill_value=0)

        precio = modelo.predict(entrada)[0]

        prediccion = round(precio, 2)

    return render_template(
        "index.html",
        prediccion=prediccion,
        info_modelo=info_modelo,
        ranking_variables=ranking_variables,
        variables_formulario=variables_formulario
    )