from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and encoders
models = {
    "depression": joblib.load("model_depression.pkl"),
    "anxiety": joblib.load("model_anxiety.pkl"),
    "stress": joblib.load("model_stress.pkl"),
}
encoders = {
    "depression": joblib.load("encoder_depression.pkl"),
    "anxiety": joblib.load("encoder_anxiety.pkl"),
    "stress": joblib.load("encoder_stress.pkl"),
}

# Define question sets
qs = {
    "depression": ["Q3A", "Q5A", "Q10A", "Q13A", "Q16A", "Q21A", "Q24A", "Q26A", "Q31A", "Q34A", "Q37A", "Q38A", "Q42A"],
    "anxiety": ["Q2A", "Q4A", "Q7A", "Q9A", "Q15A", "Q19A", "Q20A", "Q23A", "Q25A", "Q28A", "Q30A", "Q36A", "Q40A", "Q41A"],
    "stress": ["Q1A", "Q6A", "Q8A", "Q11A", "Q12A", "Q14A", "Q17A", "Q18A", "Q22A", "Q27A", "Q29A", "Q32A", "Q33A", "Q35A"],
}

# Define interpretation
interpretation = {
    "depression": {
        "normal": "No or minimal symptoms of depression.",
        "mild": "Mild depressive symptoms, may resolve on their own.",
        "moderate": "Moderate level of depression, consider talking to someone or self-care activities.",
        "severe": "Severe symptoms, professional help is strongly recommended.",
        "extremely severe": "Very severe symptoms, immediate mental health support is advised."
    },
    "anxiety": {
        "normal": "No or minimal symptoms of anxiety.",
        "mild": "Mild anxiety, manageable with lifestyle adjustment.",
        "moderate": "Moderate anxiety level, could benefit from mental health strategies.",
        "severe": "Severe anxiety, consider consulting a professional.",
        "extremely severe": "Very high anxiety level, professional support is recommended."
    },
    "stress": {
        "normal": "No or minimal stress symptoms.",
        "mild": "Mild stress, may be situational.",
        "moderate": "Moderate stress, consider stress management techniques.",
        "severe": "High stress, could affect daily functioning.",
        "extremely severe": "Extreme stress, seek professional advice."
    }
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json  # Expecting a dict of Qs: int
        result = {}

        for label, questions in qs.items():
            # Filter only questions that exist in user_input
            available_qs = [q for q in questions if q in user_input]

            if len(available_qs) < 5:
                # Skip if too few responses for prediction
                result[label] = "insufficient data"
                continue

            input_df = pd.DataFrame([[user_input[q] for q in available_qs]], columns=available_qs)

            model = models[label]
            encoder = encoders[label]

            # Align columns to model’s expected order (use intersection)
            input_df = input_df.reindex(columns=questions, fill_value=0)  # fill missing with 0 or mean

            pred = model.predict(input_df)[0]
            result[label] = encoder.inverse_transform([pred])[0]

        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
