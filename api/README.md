# Concrete Compressive Strength API

A FastAPI service that serves predictions from the real trained model
in `notebook/Concrete_Compressive_Strength_fixed.ipynb`.

## 1. Get the model files

Run the notebook fully (in Google Colab or Jupyter) until the
model-saving cell. It produces two files:

- `concrete_model.pkl`
- `concrete_scaler.pkl`

Download them and place them in this same folder, right next to
`main.py`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

It will start at: `http://localhost:8000`

## 4. Try the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cement": 300,
    "blast_furnace_slag": 0,
    "fly_ash": 0,
    "water": 190,
    "superplasticizer": 0,
    "coarse_aggregate": 1000,
    "fine_aggregate": 800,
    "age": 28
  }'
```

Expected response:

```json
{
  "predicted_strength_mpa": 28.71,
  "grade_label": "Standard structural"
}
```

Or open `http://localhost:8000/docs` to try the API through an
auto-generated interactive UI (Swagger UI) without writing curl
commands.

## 5. Connect it to the calculator page

Use `frontend/renoa-concrete-demo-local.html` instead of the version
published on claude.ai — published claude.ai pages cannot reach any
external API for security reasons. Open the local HTML file directly
from your machine and make sure the server is running on the same
port (8000).

## Deploying it for a live demo

If you'd like a live link instead of running it locally during an
interview, two free options:

- **Render.com**: push this folder as a repo to GitHub, create a new
  "Web Service", and connect it to the repo. Render detects
  `requirements.txt` and needs the Start Command set to
  `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Railway.app**: same idea, deployable straight from the CLI.

Once you have a live URL (e.g. `https://concrete-api.onrender.com`),
open `renoa-concrete-demo-local.html` and change the `API_URL` value
at the top of the script to your URL — then you can host the
calculator page anywhere (even GitHub Pages) and have it talk to a
live API without running anything locally during the demo.
