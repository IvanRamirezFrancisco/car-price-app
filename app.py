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

    if request.method == "POST":
        # Recibimos datos del formulario
        datos = {
            "symboling": int(request.form["symboling"]),
            "wheelbase": float(request.form["wheelbase"]),
            "carlength": float(request.form["carlength"]),
            "carwidth": float(request.form["carwidth"]),
            "carheight": float(request.form["carheight"]),
            "curbweight": int(request.form["curbweight"]),
            "enginesize": int(request.form["enginesize"]),
            "boreratio": float(request.form["boreratio"]),
            "stroke": float(request.form["stroke"]),
            "compressionratio": float(request.form["compressionratio"]),
            "horsepower": int(request.form["horsepower"]),
            "peakrpm": int(request.form["peakrpm"]),
            "citympg": int(request.form["citympg"]),
            "highwaympg": int(request.form["highwaympg"]),
            "fueltype": request.form["fueltype"],
            "aspiration": request.form["aspiration"],
            "doornumber": request.form["doornumber"],
            "carbody": request.form["carbody"],
            "drivewheel": request.form["drivewheel"],
            "enginelocation": request.form["enginelocation"],
            "enginetype": request.form["enginetype"],
            "cylindernumber": request.form["cylindernumber"],
            "fuelsystem": request.form["fuelsystem"]
        }

        # Convertimos a DataFrame
        entrada = pd.DataFrame([datos])

        # Convertimos categóricas igual que en entrenamiento
        entrada = pd.get_dummies(entrada, drop_first=False)

        # Ajustamos columnas para que coincidan con las usadas por el modelo
        entrada = entrada.reindex(columns=features, fill_value=0)

        # Predicción
        precio = modelo.predict(entrada)[0]

        prediccion = round(precio, 2)

    return render_template("index.html", prediccion=prediccion)

if __name__ == "__main__":
    app.run(debug=True)