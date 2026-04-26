# Model Results Web Viewer

A small Flask web app to display sample input data and prediction results from the attached dataset files.

## Files

- `app.py` - Flask application server
- `templates/index.html` - HTML view for rendering tables
- `static/style.css` - styles for the web app
- `requirements.txt` - Python dependency list

## Run locally

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   python app.py
   ```

3. Open your browser at `http://localhost:5000`

## Notes

- The app reads `sample input data.txt` only as a preview of the input schema. Predictions require pasting one full raw feature row.
- You can paste a single comma-separated row to generate a prediction using the model's expected raw input schema.
- The existing `sample predictions.txt` is shown for reference.
- Use the model file list to download the trained artifact files if needed.
