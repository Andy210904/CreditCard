# app.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import json
import sqlite3
from datetime import datetime
import io
import google.generativeai as genai
from dotenv import load_dotenv

# ==============================
# Setup & Initialization
# ==============================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "credit_risk.db")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# In-memory storage for the uploaded CSV data
client_data_store = []

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_timestamp TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        input_json TEXT NOT NULL,
        rule_final_evaluation TEXT,
        llm_final_rating TEXT,
        rule_summary TEXT,
        llm_summary TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS factors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id INTEGER NOT NULL,
        source TEXT NOT NULL, -- 'rule' or 'llm'
        factor_name TEXT NOT NULL,
        evaluation TEXT NOT NULL,
        FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
    );
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

init_db()

# ==============================
# Gemini LLM Engine
# ==============================
CLASSIFICATION_RULES = """
You are a deterministic credit risk classification engine. Your task is to:
1.  **Read** the provided JSON input.
2.  **Apply** the following 23 rules strictly. A 'High' classification means Low Credit Risk.
3.  **Handle Missing Data**: If a factor is missing from the input JSON or its value is null, classify it as 'Medium'.

**Factor Classification Rules:**
1.  `ebitda_margin_pct`: If value >= 20 -> "High" ; else if value < 5 -> "Low" ; else -> "Medium".
2.  `ebit_margin_pct`: If value >= 15 -> "High" ; else if value < 3 -> "Low" ; else -> "Medium".
3.  `debt_to_equity`: If value <= 0.5 -> "High" ; else if value >= 2.0 -> "Low" ; else -> "Medium".
4.  `interest_coverage`: If value >= 8 -> "High" ; else if value < 1.5 -> "Low" ; else -> "Medium".
5.  `dscr`: If value >= 2.5 -> "High" ; else if value < 1.0 -> "Low" ; else -> "Medium".
6.  `current_ratio`: If value >= 2.0 -> "High" ; else if value < 1.0 -> "Low" ; else -> "Medium".
7.  `quick_ratio`: If value >= 1.5 -> "High" ; else if value < 0.8 -> "Low" ; else -> "Medium".
8.  `revenue_usd_m`: If value >= 10000 -> "High" ; else if value <= 5 -> "Low" ; else -> "Medium".
9.  `revenue_cagr_3y_pct`: If value >= 12 -> "High" ; else if value < 0 -> "Low" ; else -> "Medium".
10. `years_in_operation`: If value >= 20 -> "High" ; else if value <= 3 -> "Low" ; else -> "Medium".
11. `governance_score_0_100`: If value >= 80 -> "High" ; else if value < 40 -> "Low" ; else -> "Medium".
12. `auditor_tier_code`: If value = 1 -> "High" ; else -> "Low".
13. `financials_audited_code`: If value = 1 -> "High" ; else -> "Low".
14. `esg_controversies_3y`: If value = 0 -> "High" ; else if value >= 3 -> "Low" ; else -> "Medium".
15. `country_risk_0_100`: If value <= 20 -> "High" ; else if value >= 70 -> "Low" ; else -> "Medium".
16. `industry_cyclicality_code`: If value = 0 -> "High" ; else if value = 2 -> "Low" ; else -> "Medium".
17. `fx_revenue_pct`: If value <= 10 -> "High" ; else if value >= 50 -> "Low" ; else -> "Medium".
18. `hedging_policy_code`: If value = 2 -> "High" ; else if value = 0 -> "Low" ; else -> "Medium".
19. `collateral_coverage_pct`: If value >= 150 -> "High" ; else if value < 50 -> "Low" ; else -> "Medium".
20. `covenant_quality_code`: If value = 2 -> "High" ; else if value = 0 -> "Low" ; else -> "Medium".
21. `payment_incidents_12m`: If value = 0 -> "High" ; else if value >= 3 -> "Low" ; else -> "Medium".
22. `legal_disputes_open`: If value = 0 -> "High" ; else if value >= 2 -> "Low" ; else -> "Medium".
23. `sanctions_exposure_code`: If value = 0 -> "High" ; else if value = 2 -> "Low" ; else -> "Medium".

4.  **Aggregate** the results:
    * If **>=60%** of factors are 'High' → `final_risk_rating` = **"Low risk"**
    * If **>=60%** of factors are 'Low' → `final_risk_rating` = **"High risk"**
    * Otherwise → `final_risk_rating` = **"Medium risk"**

5.  **Output** a strict JSON object with two keys: `final_risk_rating` and `factor_breakdown`.
"""

# --- Configure Gemini API ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ Warning: GEMINI_API_KEY not found. Gemini features disabled.")
else:
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini Model configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(title="Credit Risk Dashboard", version="3.0-aligned")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# ==============================
# Database Functions
# ==============================
def save_evaluation_to_db(entity_name, input_json, rule_output, llm_output):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations (evaluation_timestamp, entity_name, input_json, rule_final_evaluation, llm_final_rating, rule_summary, llm_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entity_name,
        json.dumps(input_json),
        rule_output.get("final_evaluation"),
        llm_output.get("final_risk_rating"),
        rule_output.get("summary"),
        "Evaluation generated by Gemini." if "error" not in llm_output else llm_output.get("error")
    ))
    evaluation_id = cursor.lastrowid
    if "factors" in rule_output:
        for factor in rule_output["factors"]:
            cursor.execute("INSERT INTO factors (evaluation_id, source, factor_name, evaluation) VALUES (?, 'rule', ?, ?)",
                           (evaluation_id, factor["factor"], factor["evaluation"]))
    if "factor_breakdown" in llm_output:
        for factor_name, evaluation in llm_output["factor_breakdown"].items():
            cursor.execute("INSERT INTO factors (evaluation_id, source, factor_name, evaluation) VALUES (?, 'llm', ?, ?)",
                           (evaluation_id, factor_name, evaluation))
    conn.commit()
    conn.close()
    print(f"✅ Evaluation for {entity_name} (ID: {evaluation_id}) saved.")


# ==============================
# Rule-Based Model Engine (Aligned with Gemini Prompt)
# ==============================
def apply_full_ruleset(data: dict):
    """
    Applies the definitive 23-factor rule set.
    This logic is identical to the instructions given to the Gemini model.
    """
    factors = []
    
    # Helper to evaluate rules, treating missing values as "Medium"
    def evaluate(factor_name, high_thresh, low_thresh, high_inclusive=True, low_inclusive=True):
        val = data.get(factor_name)
        if val is None: return "Medium"
        try: # Ensure value can be compared
            val = float(val)
            if (val >= high_thresh if high_inclusive else val > high_thresh): return "High"
            if (val <= low_thresh if low_inclusive else val < low_thresh): return "Low"
            return "Medium"
        except (ValueError, TypeError):
            return "Medium" # Cannot compare, so default to Medium

    # Rule Implementations
    factors.append({"factor": "ebitda_margin_pct", "evaluation": evaluate("ebitda_margin_pct", 20, 5, low_inclusive=False)})
    factors.append({"factor": "ebit_margin_pct", "evaluation": evaluate("ebit_margin_pct", 15, 3, low_inclusive=False)})
    factors.append({"factor": "debt_to_equity", "evaluation": evaluate("debt_to_equity", 2.0, 0.5, high_inclusive=False, low_inclusive=True)}) # Logic reversed for low/high
    factors.append({"factor": "interest_coverage", "evaluation": evaluate("interest_coverage", 8, 1.5, low_inclusive=False)})
    factors.append({"factor": "dscr", "evaluation": evaluate("dscr", 2.5, 1.0, low_inclusive=False)})
    factors.append({"factor": "current_ratio", "evaluation": evaluate("current_ratio", 2.0, 1.0, low_inclusive=False)})
    factors.append({"factor": "quick_ratio", "evaluation": evaluate("quick_ratio", 1.5, 0.8, low_inclusive=False)})
    factors.append({"factor": "revenue_usd_m", "evaluation": evaluate("revenue_usd_m", 10000, 5)})
    factors.append({"factor": "revenue_cagr_3y_pct", "evaluation": evaluate("revenue_cagr_3y_pct", 12, 0, low_inclusive=False)})
    factors.append({"factor": "years_in_operation", "evaluation": evaluate("years_in_operation", 20, 3)})
    factors.append({"factor": "governance_score_0_100", "evaluation": evaluate("governance_score_0_100", 80, 40, low_inclusive=False)})
    factors.append({"factor": "auditor_tier_code", "evaluation": "High" if data.get("auditor_tier_code") == 1 else "Low" if data.get("auditor_tier_code") is not None else "Medium"})
    factors.append({"factor": "financials_audited_code", "evaluation": "High" if data.get("financials_audited_code") == 1 else "Low" if data.get("financials_audited_code") is not None else "Medium"})
    val_esg = data.get('esg_controversies_3y'); factors.append({"factor": "esg_controversies_3y", "evaluation": "Medium" if val_esg is None else "High" if val_esg == 0 else "Low" if val_esg >= 3 else "Medium"})
    val_country_risk = data.get('country_risk_0_100'); factors.append({"factor": "country_risk_0_100", "evaluation": "Medium" if val_country_risk is None else "High" if val_country_risk <= 20 else "Low" if val_country_risk >= 70 else "Medium"})
    val_industry = data.get('industry_cyclicality_code'); factors.append({"factor": "industry_cyclicality_code", "evaluation": "Medium" if val_industry is None else "High" if val_industry == 0 else "Low" if val_industry == 2 else "Medium"})
    val_fx = data.get('fx_revenue_pct'); factors.append({"factor": "fx_revenue_pct", "evaluation": "Medium" if val_fx is None else "High" if val_fx <= 10 else "Low" if val_fx >= 50 else "Medium"})
    val_hedging = data.get('hedging_policy_code'); factors.append({"factor": "hedging_policy_code", "evaluation": "Medium" if val_hedging is None else "High" if val_hedging == 2 else "Low" if val_hedging == 0 else "Medium"})
    factors.append({"factor": "collateral_coverage_pct", "evaluation": evaluate("collateral_coverage_pct", 150, 50, low_inclusive=False)})
    val_covenant = data.get('covenant_quality_code'); factors.append({"factor": "covenant_quality_code", "evaluation": "Medium" if val_covenant is None else "High" if val_covenant == 2 else "Low" if val_covenant == 0 else "Medium"})
    val_payment = data.get('payment_incidents_12m'); factors.append({"factor": "payment_incidents_12m", "evaluation": "Medium" if val_payment is None else "High" if val_payment == 0 else "Low" if val_payment >= 3 else "Medium"})
    val_legal = data.get('legal_disputes_open'); factors.append({"factor": "legal_disputes_open", "evaluation": "Medium" if val_legal is None else "High" if val_legal == 0 else "Low" if val_legal >= 2 else "Medium"})
    val_sanctions = data.get('sanctions_exposure_code'); factors.append({"factor": "sanctions_exposure_code", "evaluation": "Medium" if val_sanctions is None else "High" if val_sanctions == 0 else "Low" if val_sanctions == 2 else "Medium"})
    
    # Final Evaluation Calculation
    total_factors = len(factors)
    high_count = sum(1 for f in factors if f["evaluation"] == "High")
    low_count = sum(1 for f in factors if f["evaluation"] == "Low")
    
    final_evaluation = "Medium risk"
    if total_factors > 0:
        if (high_count / total_factors) >= 0.6: final_evaluation = "Low risk"
        elif (low_count / total_factors) >= 0.6: final_evaluation = "High risk"

    summary = "Evaluation based on the implemented 23-factor rule engine."
    return {"factors": factors, "summary": summary, "final_evaluation": final_evaluation}


# ==============================
# Gemini API Call Function
# ==============================
async def get_gemini_classification(data: dict):
    if not api_key: return {"error": "Gemini model is not configured. Check API key."}
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=CLASSIFICATION_RULES)
        clean_data = {k: v for k, v in data.items() if v is not None}
        prompt = f"Classify the credit risk based on this JSON data:\n{json.dumps(clean_data, indent=2)}"
        response = await model.generate_content_async(
            [prompt],
            generation_config={"response_mime_type": "application/json", "temperature": 0}
        )
        return json.loads(response.text)
    except Exception as e:
        raw_text = "N/A"
        if 'response' in locals() and hasattr(response, 'text'): raw_text = response.text
        print(f"Gemini API Error: {e}. Raw Response: {raw_text}")
        return {"error": f"Gemini API error: {e}", "raw_response": raw_text}


# ==============================
# FastAPI Routes
# ==============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "clients": client_data_store})

@app.post("/upload_csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    global client_data_store
    context = {"request": request}
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if "entity_name" not in df.columns:
            context["error_message"] = "CSV must contain an 'entity_name' column."
            return templates.TemplateResponse("index.html", context)
        df = df.where(pd.notnull(df), None)
        client_data_store = df.to_dict(orient='records')
        context["clients"] = client_data_store
    except Exception as e:
        context["error_message"] = f"Error processing CSV file: {e}"
    return templates.TemplateResponse("index.html", context)

@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate_client(request: Request, row_index: int = Form(...)):
    context = {"request": request}
    try:
        input_data = client_data_store[row_index]
        entity_name = input_data.get("entity_name", "Unknown Client")
        context["entity_name"] = entity_name
        
        rule_output = apply_full_ruleset(input_data)
        context["rule_output"] = rule_output

        llm_output = await get_gemini_classification(input_data)
        context["llm_output"] = llm_output
        
        save_evaluation_to_db(entity_name, input_data, rule_output, llm_output)
        
    except IndexError:
        context["error_message"] = "Invalid row index. Please re-upload the file."
        return templates.TemplateResponse("index.html", context)
    except Exception as e:
        context["error_message"] = f"An evaluation error occurred: {e}"
    
    return templates.TemplateResponse("gauge.html", context)

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, evaluation_timestamp, entity_name, rule_final_evaluation, llm_final_rating FROM evaluations ORDER BY evaluation_timestamp DESC")
    evaluations = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse("history.html", {"request": request, "evaluations": evaluations})
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# ...existing code...

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)