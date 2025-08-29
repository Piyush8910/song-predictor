from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        explicit = int(request.form['explicit'])  # <-- fixed name & type

        # prepare features
        features = np.array([[danceability, energy, explicit]])
        features = scaler.transform(features)

        # prediction
        pred = model.predict(features)[0]
        pred = int(max(0, min(100, pred)))  # clamp between 0–100

        if pred <= 30:
            status = "Not very popular 💤"
        elif pred <= 60:
            status = "Moderately popular 🎶"
        elif pred <= 80:
            status = "Quite popular 🔥"
        else:
            status = "Hit song! 🚀"

        return render_template("index.html", prediction=f"{pred}/100 → {status}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)

