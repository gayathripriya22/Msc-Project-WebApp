from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import csv
import os
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "sample_input": "sample input data.txt",
    "sample_predictions": "sample predictions.txt",
}
MODEL_FILES = [
    "label_encoder.pkl",
    "onehot_encoder.pkl",
    "random_forest_model.pkl",
    "xgboost_model.pkl",
]
RF_MODEL = None
XGB_MODEL = None
LABEL_ENCODER = None
ONEHOT_ENCODER = None
RAW_FEATURE_NAMES = []
NUMERIC_FEATURE_NAMES = []
CATEGORICAL_FEATURES = []
MODEL_LOAD_ERRORS = {}


def read_csv_data(source, max_rows=None):
    if hasattr(source, "read"):
        csvfile = source
        csvfile.seek(0)
        close_file = False
    else:
        csvfile = open(os.path.join(BASE_DIR, source), newline="", encoding="utf-8")
        close_file = True

    rows = []
    columns = []
    try:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames or []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    except FileNotFoundError:
        columns = []
        rows = []
    finally:
        if close_file:
            csvfile.close()

    return columns, rows


def load_models():
    global RF_MODEL, XGB_MODEL, LABEL_ENCODER, ONEHOT_ENCODER
    global RAW_FEATURE_NAMES, NUMERIC_FEATURE_NAMES, CATEGORICAL_FEATURES

    if RF_MODEL is not None or XGB_MODEL is not None or LABEL_ENCODER is not None or ONEHOT_ENCODER is not None:
        return

    MODEL_LOAD_ERRORS.clear()

    def load_artifact(filename):
        try:
            return joblib.load(os.path.join(BASE_DIR, filename))
        except Exception as exc:
            MODEL_LOAD_ERRORS[filename] = str(exc)
            return None

    ONEHOT_ENCODER = load_artifact("onehot_encoder.pkl")
    RF_MODEL = load_artifact("random_forest_model.pkl")
    XGB_MODEL = load_artifact("xgboost_model.pkl")
    LABEL_ENCODER = load_artifact("label_encoder.pkl")

    if RF_MODEL is not None:
        RAW_FEATURE_NAMES = list(RF_MODEL.feature_names_in_)
        if ONEHOT_ENCODER is not None:
            CATEGORICAL_FEATURES = list(ONEHOT_ENCODER.feature_names_in_)
            NUMERIC_FEATURE_NAMES = []
            for name in RAW_FEATURE_NAMES:
                if any(name.startswith(prefix + "_") for prefix in CATEGORICAL_FEATURES):
                    break
                NUMERIC_FEATURE_NAMES.append(name)


def get_model_status():
    load_models()
    return {
        "Random Forest": "loaded" if RF_MODEL is not None else MODEL_LOAD_ERRORS.get("random_forest_model.pkl", "missing"),
        "XGBoost": "loaded" if XGB_MODEL is not None else MODEL_LOAD_ERRORS.get("xgboost_model.pkl", "missing"),
        "Label encoder": "loaded" if LABEL_ENCODER is not None else MODEL_LOAD_ERRORS.get("label_encoder.pkl", "missing"),
    }


def prepare_feature_vector(row):
    load_models()
    if RF_MODEL is None and XGB_MODEL is None:
        raise RuntimeError("No model is available to generate predictions.")
    if ONEHOT_ENCODER is None:
        raise RuntimeError("OneHotEncoder is required to prepare input features.")

    feature_values = []
    for name in NUMERIC_FEATURE_NAMES:
        if name not in row:
            raise ValueError(f"Missing required field '{name}' in pasted row.")
        text = str(row[name]).strip()
        if text == "":
            feature_values.append(0.0)
            continue
        try:
            feature_values.append(float(text))
        except ValueError:
            raise ValueError(f"Field '{name}' must be numeric, but got '{text}'.")

    categorical_values = []
    for name in CATEGORICAL_FEATURES:
        if name not in row:
            raise ValueError(f"Missing required field '{name}' in pasted row.")
        categorical_values.append(str(row[name]).strip())

    encoded = ONEHOT_ENCODER.transform([categorical_values])
    try:
        encoded = encoded.toarray()[0]
    except AttributeError:
        encoded = encoded[0]

    feature_values.extend(encoded.tolist())
    return feature_values


def decode_prediction(prediction):
    if LABEL_ENCODER is None:
        return prediction
    try:
        return LABEL_ENCODER.inverse_transform([prediction])[0]
    except Exception:
        return prediction


def build_prediction_rows(input_rows):
    load_models()
    prediction_rows = []
    error_message = None

    if RF_MODEL is None and XGB_MODEL is None:
        error_message = "No model files were available to generate predictions."
        return prediction_rows, error_message

    for row in input_rows:
        rf_prediction = None
        xgb_prediction = None
        try:
            feature_vector = prepare_feature_vector(row)
        except Exception as exc:
            rf_prediction = f"Error: {exc}"
            xgb_prediction = f"Error: {exc}"
            row_copy = row.copy()
            row_copy["Random Forest Prediction"] = rf_prediction
            row_copy["XGBoost Prediction"] = xgb_prediction
            prediction_rows.append(row_copy)
            continue

        if RF_MODEL is not None:
            try:
                rf_raw = RF_MODEL.predict([feature_vector])[0]
                rf_prediction = decode_prediction(rf_raw)
            except Exception as exc:
                rf_prediction = f"Error: {exc}"

        if XGB_MODEL is not None:
            try:
                xgb_raw = XGB_MODEL.predict([feature_vector])[0]
                xgb_prediction = decode_prediction(xgb_raw)
            except Exception as exc:
                xgb_prediction = f"Error: {exc}"

        row_copy = row.copy()
        row_copy["Random Forest Prediction"] = rf_prediction
        row_copy["XGBoost Prediction"] = xgb_prediction
        prediction_rows.append(row_copy)

    return prediction_rows, error_message


def get_expected_input_columns():
    load_models()
    if not NUMERIC_FEATURE_NAMES or not CATEGORICAL_FEATURES:
        return []
    return NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURES


def parse_pasted_row(row_text):
    reader = csv.reader([row_text.strip()])
    values = next(reader, [])
    if not values:
        raise ValueError("No data found in pasted row.")

    values = [value.strip() for value in values]
    expected_columns = get_expected_input_columns()
    if not expected_columns:
        raise RuntimeError("Model preprocessing artifacts are missing.")

    if len(values) == len(expected_columns) + 1:
        values = values[1:]

    if len(values) != len(expected_columns):
        raise ValueError(
            f"Expected {len(expected_columns)} values, but got {len(values)}. "
            f"The model requires a full raw feature row with the expected input schema."
        )

    return [dict(zip(expected_columns, values))]


def render_main_page(
    input_columns,
    input_rows,
    pred_columns,
    pred_rows,
    prediction_rows,
    prediction_error,
    submitted_source=None,
):
    prediction_columns = input_columns + ["Random Forest Prediction", "XGBoost Prediction"]
    return render_template(
        "index.html",
        input_columns=input_columns,
        input_rows=input_rows,
        pred_columns=pred_columns,
        pred_rows=pred_rows,
        prediction_columns=prediction_columns,
        prediction_rows=prediction_rows,
        prediction_error=prediction_error,
        model_files=MODEL_FILES,
        model_status=get_model_status(),
        submitted_source=submitted_source,
    )


@app.route("/", methods=["GET"])
def index():
    input_columns, input_rows = read_csv_data(FILES["sample_input"], max_rows=100)
    pred_columns, pred_rows = read_csv_data(FILES["sample_predictions"], max_rows=100)
    prediction_rows = []
    prediction_error = None
    return render_main_page(
        input_columns,
        input_rows,
        pred_columns,
        pred_rows,
        prediction_rows,
        prediction_error,
    )


@app.route("/predict", methods=["POST"])
def predict_row():
    row_text = request.form.get("rowdata", "").strip()
    if not row_text:
        flash("Please paste one comma-separated row of data.")
        return redirect(url_for("index"))

    try:
        input_rows = parse_pasted_row(row_text)
        prediction_rows, prediction_error = build_prediction_rows(input_rows)
        pred_columns, pred_rows = read_csv_data(FILES["sample_predictions"], max_rows=100)
        expected_columns = get_expected_input_columns()

        return render_main_page(
            expected_columns,
            input_rows,
            pred_columns,
            pred_rows,
            prediction_rows,
            prediction_error,
            submitted_source="pasted row",
        )
    except Exception as exc:
        flash(str(exc))
        return redirect(url_for("index"))


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(BASE_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
