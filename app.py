from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model once when the app starts
# Ensure salary_predict_model.pkl is in the same folder as this file!
model = joblib.load("salary_predict_model.pkl")

@app.route("/")
def home():
    """Landing page for the Salary Prediction API"""
    return (
        "<h1>Salary Prediction API</h1>"
        "<p>BAIS:3300 - Digital Product Development</p>"
        "<p>Ryder Rietz</p>"
    )

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict salary based on input JSON payload
    Expected keys: age, gender, country, highest_deg, coding_exp, title, company_size
    """
    try:
        data = request.get_json()

        required_fields = [
            "age",
            "gender",
            "country",
            "highest_deg",
            "coding_exp",
            "title",
            "company_size",
        ]
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Ensure correct order and type
        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["country"]),
            int(data["highest_deg"]),
            int(data["coding_exp"]),
            int(data["title"]),
            int(data["company_size"]),
        ]

        prediction = model.predict([features])[0]

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Standard Flask port
    app.run(host="0.0.0.0", port=8080, debug=True)
    