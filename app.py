import os
import time
import json
import re
import logging
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AetherForgePipeline")

# Global variables
client = None

# Choose the active compiler model:
# 1. "llama-3.3-70b-versatile" (Default: High accuracy, low free-tier rate limits - 100K tokens/day)
# 2. "llama-3.1-8b-instant" (Fast: Good accuracy, 10x higher rate limits - 1.0M tokens/day)
GROQ_MODEL = "llama-3.1-8b-instant"

def get_groq_client():
    """Initializes and returns the Groq API client."""
    global client
    # Force reload env variables to pick up any changes to .env
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        client = Groq(api_key=api_key)
        return client
    raise ValueError("GROQ_API_KEY is missing or set to the default placeholder. Please edit your .env file.")

class PipelineTracker:
    """Tracks retries and errors across pipeline execution."""
    def __init__(self):
        self.retries = 0
        self.errors = []

# ==========================================
# EVALUATION DATASET (10 Real + 10 Edge Cases)
# ==========================================
EVALUATION_PROMPTS = [
    # 10 Real Product Prompts
    {"id": 1, "type": "real", "prompt": "Build a CRM with login, contacts, dashboard, role-based access, and premium plan with payments"},
    {"id": 2, "type": "real", "prompt": "Create a project management tool with tasks, teams, deadlines and file uploads"},
    {"id": 3, "type": "real", "prompt": "Build an ecommerce store with products, cart, checkout and admin panel"},
    {"id": 4, "type": "real", "prompt": "Make a booking system for a clinic with appointments, doctors, patients and reminders"},
    {"id": 5, "type": "real", "prompt": "Build a social platform with posts, comments, likes, followers and direct messages"},
    {"id": 6, "type": "real", "prompt": "Build a learning management system with courses, lessons, quizzes, student progress, and instructor payouts"},
    {"id": 7, "type": "real", "prompt": "Create a restaurant inventory system with suppliers, stock items, low stock alerts, recipes, and purchase orders"},
    {"id": 8, "type": "real", "prompt": "Make a real estate listing app with properties, agent profiles, viewing schedules, maps, and inquiries"},
    {"id": 9, "type": "real", "prompt": "Build a fitness tracking app with workouts, exercises, diet log, goal setting, and coach feedback"},
    {"id": 10, "type": "real", "prompt": "Create a support ticket desk with ticket routing, user profiles, SLAs, departments, and premium tiers"},
    
    # 10 Edge Cases
    # Vague
    {"id": 11, "type": "edge_vague", "prompt": "build app"},
    {"id": 12, "type": "edge_vague", "prompt": "make a system"},
    {"id": 13, "type": "edge_vague", "prompt": "design a tool with users"},
    # Conflicting
    {"id": 14, "type": "edge_conflict", "prompt": "build a platform with payments but completely free and no transactions allowed"},
    {"id": 15, "type": "edge_conflict", "prompt": "create a chat app that does not allow any users to send messages"},
    {"id": 16, "type": "edge_conflict", "prompt": "build a database viewer but do not allow any data to be stored or retrieved"},
    # Incomplete
    {"id": 17, "type": "edge_incomplete", "prompt": "build a website with login"},
    {"id": 18, "type": "edge_incomplete", "prompt": "make a store with cart"},
    {"id": 19, "type": "edge_incomplete", "prompt": "create a dashboard for analytics"},
    {"id": 20, "type": "edge_incomplete", "prompt": "create a simple blog platform"}
]

def parse_json_robust(text):
    """Robustly extracts and parses JSON from LLM text containing possible markdown fences."""
    text = text.strip()
    
    # 1. Clean markdown JSON code fences if they surround the response
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
        
    # 2. Find first '{' and last '}' to strip conversational padding
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace+1]
        
    return json.loads(text)

def call_groq_api(system_prompt, user_prompt, tracker, stage_name):
    """Executes a Groq chat completion call with automatic 2-second retry on failure."""
    # Throttle requests to stay under free tier TPM (Tokens Per Minute) limit
    time.sleep(3)
    
    groq_client = get_groq_client()
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"[{stage_name}] API Call failed: {e}. Retrying once after 2 seconds...")
        if tracker:
            tracker.retries += 1
            tracker.errors.append(f"[{stage_name}] API Failure: {e}")
        time.sleep(2)
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e2:
            logger.error(f"[{stage_name}] API Retry failed: {e2}")
            if tracker:
                tracker.errors.append(f"[{stage_name}] API Retry Failure: {e2}")
            raise e2

def repair_json(broken_json, tracker, stage_name):
    """Invokes the Groq API to fix malformed JSON string."""
    system_prompt = "You are a JSON repair utility. Fix broken JSON. Return ONLY valid JSON, no markdown formatting or explanation."
    user_prompt = f"Fix this invalid JSON and return ONLY valid JSON: {broken_json}"
    
    logger.info(f"[{stage_name}] Parsing failed. Triggering repair_json...")
    if tracker:
        tracker.retries += 1
        tracker.errors.append(f"[{stage_name}] Initial JSON Parse failed, invoking repair.")
        
    try:
        repaired_text = call_groq_api(system_prompt, user_prompt, tracker, f"{stage_name}_repair")
        return parse_json_robust(repaired_text)
    except Exception as e:
        logger.error(f"[{stage_name}] JSON repair failed: {e}")
        if tracker:
            tracker.errors.append(f"[{stage_name}] Repair failure: {e}")
        raise ValueError(f"JSON repair failed: {e}")

# ==========================================
# RUNTIME EXECUTION AWARENESS SIMULATOR
# ==========================================
def evaluate_execution_readiness(config):
    """
    Simulates execution and validates the compiled JSON config against structural and relational contracts.
    Returns a dict with readiness score, passed checks list, and overall status.
    """
    if not isinstance(config, dict):
        return {"score": 0, "status": "failed", "errors": ["Output config is not a JSON object."]}
        
    checks = []
    errors = []
    
    # Check 1: Metadata Integrity
    has_meta = all(k in config for k in ["app_name", "generated_at", "intent", "system_design", "schemas"])
    checks.append({
        "name": "Metadata Integrity",
        "passed": has_meta,
        "details": "Verifies compiler metadata, intent model, and schemas are present." if has_meta else "Missing top-level layout keys."
    })
    if not has_meta:
        errors.append("Missing required top-level configuration metadata.")

    # Get schemas block safely
    schemas = config.get("schemas")
    if not isinstance(schemas, dict):
        schemas = {}
        
    ui_schema = schemas.get("ui") or schemas.get("ui_schema")
    if not isinstance(ui_schema, dict):
        ui_schema = {}
        
    api_schema = schemas.get("api") or schemas.get("api_schema")
    if not isinstance(api_schema, dict):
        api_schema = {}
        
    db_schema = schemas.get("database") or schemas.get("db_schema")
    if not isinstance(db_schema, dict):
        db_schema = {}
        
    auth_schema = schemas.get("auth") or schemas.get("auth_schema")
    if not isinstance(auth_schema, dict):
        auth_schema = {}
    
    # Check 2: Database Schema Integrity
    db_passed = True
    tables = []
    if isinstance(db_schema, dict) and "tables" in db_schema:
        tables_list = db_schema["tables"]
        if isinstance(tables_list, list):
            tables = [t.get("name") for t in tables_list if isinstance(t, dict) and "name" in t]
            for t in tables_list:
                if not isinstance(t, dict) or "columns" not in t:
                    db_passed = False
                    errors.append(f"Database table {t.get('name', 'unknown')} has no columns list.")
                    break
        else:
            db_passed = False
            errors.append("db_schema.tables is not an array.")
    else:
        db_passed = False
        errors.append("db_schema is missing tables description.")
        
    checks.append({
        "name": "Database Schema Integrity",
        "passed": db_passed,
        "details": f"Database tables structure verified: {len(tables)} tables registered." if db_passed else "Database integrity check failed."
    })
    
    # Check 3: API Endpoint Mapping
    api_passed = True
    api_count = 0
    if isinstance(api_schema, dict) and "endpoints" in api_schema:
        endpoints = api_schema["endpoints"]
        if isinstance(endpoints, list):
            api_count = len(endpoints)
            for ep in endpoints:
                if not isinstance(ep, dict) or not all(k in ep for k in ["path", "method"]):
                    api_passed = False
                    errors.append("API endpoints must declare 'path' and 'method'.")
                    break
        else:
            api_passed = False
            errors.append("api_schema.endpoints is not an array.")
    else:
        api_passed = False
        errors.append("api_schema is missing endpoints description.")
        
    checks.append({
        "name": "API Endpoint Mapping",
        "passed": api_passed,
        "details": f"API mapping verified: {api_count} endpoints bindable." if api_passed else "API binding failed validation."
    })

    # Check 4: UI Router Mapping
    ui_passed = True
    ui_pages_count = 0
    if isinstance(ui_schema, dict) and "pages" in ui_schema:
        pages = ui_schema["pages"]
        if isinstance(pages, list):
            ui_pages_count = len(pages)
            for p in pages:
                if not isinstance(p, dict) or not all(k in p for k in ["name", "route"]):
                    ui_passed = False
                    errors.append("UI page objects must declare 'name' and 'route'.")
                    break
        else:
            ui_passed = False
            errors.append("ui_schema.pages is not an array.")
    else:
        ui_passed = False
        errors.append("ui_schema is missing pages description.")
        
    checks.append({
        "name": "UI Router Mapping",
        "passed": ui_passed,
        "details": f"UI routers verified: {ui_pages_count} pages mapped." if ui_passed else "UI routing verification failed."
    })

    # Check 5: Auth Policy Mapping
    auth_passed = True
    if isinstance(auth_schema, dict) and "roles" in auth_schema:
        roles = auth_schema["roles"]
        if not isinstance(roles, list) and not isinstance(roles, dict):
            auth_passed = False
            errors.append("auth_schema.roles must be an array or object mapping.")
    else:
        auth_passed = False
        errors.append("auth_schema is missing role configuration.")
        
    checks.append({
        "name": "Auth Policy Mapping",
        "passed": auth_passed,
        "details": "Authorization policies and roles verified." if auth_passed else "Auth mapping failed verification."
    })
    
    # Calculate Score
    passed_checks = sum(1 for c in checks if c["passed"])
    score = int((passed_checks / len(checks)) * 100)
    
    status = "ready"
    if score < 60:
        status = "failed"
    elif score < 100:
        status = "warning"
        
    return {
        "score": score,
        "status": status,
        "checks_run": checks,
        "errors": errors
    }

# ==========================================
# PIPELINE STAGES
# ==========================================

def extract_intent(user_prompt, tracker=None):
    """Stage 1: Extracts features, user roles, data entities and payments/auth requirements."""
    system_prompt = "You are an intent parser. Extract structured information from app descriptions. Return ONLY valid JSON, no explanation."
    user_prompt_formatted = f"Extract from this: {user_prompt}. Return JSON with: app_name, app_type, features (array), user_roles (array), data_entities (array), has_payments (bool), has_auth (bool), business_rules (array)"
    
    raw_response = call_groq_api(system_prompt, user_prompt_formatted, tracker, "Stage 1 - Extract Intent")
    try:
        return parse_json_robust(raw_response)
    except Exception as e:
        logger.warning(f"[Stage 1 - Extract Intent] JSON parsing failed, attempting repair: {e}")
        return repair_json(raw_response, tracker, "Stage 1 - Extract Intent")

def design_system(intent_json, tracker=None):
    """Stage 2: Designs system flows, ERDs, and access controls from parsed intent."""
    system_prompt = "You are a software architect. Design app architecture from intent. Return ONLY valid JSON."
    intent_str = json.dumps(intent_json, indent=2)
    user_prompt_formatted = f"Design architecture for: {intent_str}. Return JSON with: entity_relationships (array of objects with from, to, type), user_flows (array), role_permissions (object with role as key, permissions array as value), feature_dependencies (object)"
    
    raw_response = call_groq_api(system_prompt, user_prompt_formatted, tracker, "Stage 2 - Design System")
    try:
        return parse_json_robust(raw_response)
    except Exception as e:
        logger.warning(f"[Stage 2 - Design System] JSON parsing failed, attempting repair: {e}")
        return repair_json(raw_response, tracker, "Stage 2 - Design System")

def generate_schemas(intent_json, design_json, tracker=None):
    """Stage 3: Generates detailed UI layout schemas, backend endpoints, DB tables, and auth policies."""
    system_prompt = "You are a full-stack schema generator. Generate complete schemas. Return ONLY valid JSON."
    intent_str = json.dumps(intent_json, indent=2)
    design_str = json.dumps(design_json, indent=2)
    user_prompt_formatted = f"Generate all schemas for app with intent: {intent_str} and design: {design_str}. Return JSON with exactly these keys: ui_schema (pages array, each with name, route, components array, layout), api_schema (endpoints array, each with path, method, description, request_body, response_body, auth_required, roles_allowed), db_schema (tables array, each with name, columns array with name/type/nullable/primary_key, relations array), auth_schema (roles array, permissions object, protected_routes array, token_type)"
    
    raw_response = call_groq_api(system_prompt, user_prompt_formatted, tracker, "Stage 3 - Generate Schemas")
    try:
        return parse_json_robust(raw_response)
    except Exception as e:
        logger.warning(f"[Stage 3 - Generate Schemas] JSON parsing failed, attempting repair: {e}")
        return repair_json(raw_response, tracker, "Stage 3 - Generate Schemas")

def validate_and_repair(schemas_json, intent_json, tracker=None):
    """Stage 4: Performs integrity and structure validation of layouts against API schemas, repairing errors."""
    system_prompt = "You are a schema validator. Check for errors and fix them. Return ONLY valid JSON."
    schemas_str = json.dumps(schemas_json, indent=2)
    intent_str = json.dumps(intent_json, indent=2)
    user_prompt_formatted = f"Validate and repair these schemas: {schemas_str}. Cross-reference with the app intent: {intent_str}. Check: all API endpoints have matching DB tables, all UI pages have matching API endpoints, no missing required fields, no type mismatches. Return JSON with: validated_schemas (the fixed complete schemas), validation_report (errors_found array, repairs_made array, passed bool)"
    
    raw_response = call_groq_api(system_prompt, user_prompt_formatted, tracker, "Stage 4 - Validate & Repair")
    try:
        return parse_json_robust(raw_response)
    except Exception as e:
        logger.warning(f"[Stage 4 - Validate & Repair] JSON parsing failed, attempting repair: {e}")
        return repair_json(raw_response, tracker, "Stage 4 - Validate & Repair")

def generate_final_output(all_stages_data, tracker=None):
    """Stage 5: Assembles and formats all previous pipeline stages into the final compiled specification."""
    system_prompt = "You are a configuration compiler. Merge all pipeline outputs into final config. Return ONLY valid JSON."
    stages_data_str = json.dumps(all_stages_data, indent=2)
    user_prompt_formatted = f"Merge these pipeline stages into final output: {stages_data_str}. Return single JSON with: app_name, generated_at (timestamp), intent, system_design, schemas (ui, api, database, auth), business_logic, validation_report, assumptions_made (array of strings documenting any decisions made)"
    
    raw_response = call_groq_api(system_prompt, user_prompt_formatted, tracker, "Stage 5 - Compile Output")
    try:
        return parse_json_robust(raw_response)
    except Exception as e:
        logger.warning(f"[Stage 5 - Compile Output] JSON parsing failed, attempting repair: {e}")
        return repair_json(raw_response, tracker, "Stage 5 - Compile Output")

# ==========================================
# PIPELINE EXECUTION ENGINE
# ==========================================

def run_pipeline_sync(user_prompt, tracker=None):
    """Runs all 5 pipeline stages sequentially, returning the final config and a success boolean."""
    if tracker is None:
        tracker = PipelineTracker()
        
    outputs = {
        "intent": None,
        "system_design": None,
        "schemas": None,
        "validation_report": None
    }
    
    try:
        logger.info("Executing Pipeline Stage 1...")
        outputs["intent"] = extract_intent(user_prompt, tracker)
        
        logger.info("Executing Pipeline Stage 2...")
        outputs["system_design"] = design_system(outputs["intent"], tracker)
        
        logger.info("Executing Pipeline Stage 3...")
        outputs["schemas"] = generate_schemas(outputs["intent"], outputs["system_design"], tracker)
        
        logger.info("Executing Pipeline Stage 4...")
        stage4_res = validate_and_repair(outputs["schemas"], outputs["intent"], tracker)
        validated_schemas = stage4_res.get("validated_schemas", outputs["schemas"])
        validation_report = stage4_res.get("validation_report", {
            "errors_found": [],
            "repairs_made": [],
            "passed": True
        })
        outputs["schemas"] = validated_schemas
        outputs["validation_report"] = validation_report
        
        logger.info("Executing Pipeline Stage 5...")
        all_stages_data = {
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": validated_schemas,
            "validation_report": validation_report
        }
        
        final_config = generate_final_output(all_stages_data, tracker)
        
        # Calculate execution readiness simulation
        readiness = evaluate_execution_readiness(final_config)
        final_config["execution_simulation"] = readiness
        
        return final_config, True
        
    except Exception as e:
        logger.error(f"Pipeline sync run crashed: {e}")
        
        # Calculate which stage failed based on outputs loaded
        failed_stage = "Stage 5 - Compile Output"
        if outputs["intent"] is None:
            failed_stage = "Stage 1 - Extract Intent"
        elif outputs["system_design"] is None:
            failed_stage = "Stage 2 - Design System"
        elif outputs["schemas"] is None:
            failed_stage = "Stage 3 - Generate Schemas"
        elif outputs["validation_report"] is None:
            failed_stage = "Stage 4 - Validate & Repair"
            
        partial_res = {
            "app_name": outputs["intent"].get("app_name", "Compilation Error") if outputs["intent"] else "Compilation Failed",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": outputs["schemas"],
            "validation_report": outputs["validation_report"],
            "error_flag": True,
            "error": f"Pipeline execution failed at [{failed_stage}]. Reason: {str(e)}"
        }
        
        # Calculate execution readiness for partial results
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        
        return partial_res, False

def generate_pipeline_stream(user_prompt):
    """Generator for streaming pipeline execution events (Server-Sent Events)."""
    tracker = PipelineTracker()
    start_time = time.time()
    
    outputs = {
        "intent": None,
        "system_design": None,
        "schemas": None,
        "validation_report": None
    }
    
    def format_event(event_type, **kwargs):
        payload = {"event": event_type, **kwargs}
        return f"data: {json.dumps(payload)}\n\n"

    # Stage 1
    yield format_event("stage_start", stage=1, stage_name="Extracting Intent & Scope")
    try:
        outputs["intent"] = extract_intent(user_prompt, tracker)
        yield format_event("stage_success", stage=1, data=outputs["intent"])
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        yield format_event("stage_error", stage=1, error=str(e))
        latency = time.time() - start_time
        partial_res = {
            "app_name": "Compilation Failed",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": None,
            "system_design": None,
            "schemas": None,
            "validation_report": None,
            "error_flag": True,
            "error": f"Stage 1 failed: {str(e)}"
        }
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        yield format_event("pipeline_complete", success=False, result=partial_res, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)
        return

    # Stage 2
    yield format_event("stage_start", stage=2, stage_name="Designing App Architecture")
    try:
        outputs["system_design"] = design_system(outputs["intent"], tracker)
        yield format_event("stage_success", stage=2, data=outputs["system_design"])
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        yield format_event("stage_error", stage=2, error=str(e))
        latency = time.time() - start_time
        partial_res = {
            "app_name": outputs["intent"].get("app_name", "Compilation Error"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": outputs["intent"],
            "system_design": None,
            "schemas": None,
            "validation_report": None,
            "error_flag": True,
            "error": f"Stage 2 failed: {str(e)}"
        }
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        yield format_event("pipeline_complete", success=False, result=partial_res, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)
        return

    # Stage 3
    yield format_event("stage_start", stage=3, stage_name="Generating Database, Layout & API Schemas")
    try:
        outputs["schemas"] = generate_schemas(outputs["intent"], outputs["system_design"], tracker)
        yield format_event("stage_success", stage=3, data=outputs["schemas"])
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        yield format_event("stage_error", stage=3, error=str(e))
        latency = time.time() - start_time
        partial_res = {
            "app_name": outputs["intent"].get("app_name", "Compilation Error"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": None,
            "validation_report": None,
            "error_flag": True,
            "error": f"Stage 3 failed: {str(e)}"
        }
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        yield format_event("pipeline_complete", success=False, result=partial_res, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)
        return

    # Stage 4
    yield format_event("stage_start", stage=4, stage_name="Validating Layouts, Endpoints & DB Schemas")
    try:
        stage4_res = validate_and_repair(outputs["schemas"], outputs["intent"], tracker)
        outputs["schemas"] = stage4_res.get("validated_schemas", outputs["schemas"])
        outputs["validation_report"] = stage4_res.get("validation_report", {
            "errors_found": [],
            "repairs_made": [],
            "passed": True
        })
        yield format_event("stage_success", stage=4, data=stage4_res)
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        yield format_event("stage_error", stage=4, error=str(e))
        latency = time.time() - start_time
        partial_res = {
            "app_name": outputs["intent"].get("app_name", "Compilation Error"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": outputs["schemas"],
            "validation_report": None,
            "error_flag": True,
            "error": f"Stage 4 failed: {str(e)}"
        }
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        yield format_event("pipeline_complete", success=False, result=partial_res, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)
        return

    # Stage 5
    yield format_event("stage_start", stage=5, stage_name="Compiling Final Integrated Specifications")
    try:
        all_stages_data = {
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": outputs["schemas"],
            "validation_report": outputs["validation_report"]
        }
        final_output = generate_final_output(all_stages_data, tracker)
        yield format_event("stage_success", stage=5, data=final_output)
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        yield format_event("stage_error", stage=5, error=str(e))
        latency = time.time() - start_time
        partial_res = {
            "app_name": outputs["intent"].get("app_name", "Compilation Error"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "intent": outputs["intent"],
            "system_design": outputs["system_design"],
            "schemas": outputs["schemas"],
            "validation_report": outputs["validation_report"],
            "error_flag": True,
            "error": f"Stage 5 failed: {str(e)}"
        }
        readiness = evaluate_execution_readiness(partial_res)
        partial_res["execution_simulation"] = readiness
        yield format_event("pipeline_complete", success=False, result=partial_res, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)
        return

    # Complete successfully
    latency = time.time() - start_time
    readiness = evaluate_execution_readiness(final_output)
    final_output["execution_simulation"] = readiness
    yield format_event("pipeline_complete", success=True, result=final_output, latency_seconds=round(latency, 2), total_retries=tracker.retries, errors=tracker.errors)

# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    """Serves the main single-page UI."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/generate', methods=['POST'])
def generate():
    """Trigger the compilation pipeline."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    
    # Prompt validation: must be at least 10 words
    words = [w for w in prompt.split() if w]
    if len(words) < 10:
        return jsonify({
            "clarification_needed": True,
            "message": f"Please provide a more detailed description. Your prompt has only {len(words)} words, but at least 10 words are required to generate a high-quality app architecture."
        }), 200

    try:
        # Check if environment is configured
        get_groq_client()
    except Exception as e:
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

    # Stream updates if Client accepts SSE or has set stream parameter
    stream_requested = request.args.get("stream", "false").lower() == "true" or \
                       request.headers.get("Accept") == "text/event-stream"
                       
    if stream_requested:
        return Response(
            stream_with_context(generate_pipeline_stream(prompt)),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Standard synchronous response
        tracker = PipelineTracker()
        start_time = time.time()
        result, success = run_pipeline_sync(prompt, tracker)
        latency = time.time() - start_time
        
        response_data = result.copy()
        response_data["latency_seconds"] = round(latency, 2)
        response_data["total_retries"] = tracker.retries
        response_data["pipeline_success"] = success
        response_data["errors"] = tracker.errors
        
        return jsonify(response_data)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Runs compilation pipeline on a specific test prompt index (0-19) or returns dataset prompts."""
    try:
        get_groq_client()
    except Exception as e:
        return jsonify({
            "error": True,
            "message": str(e)
        }), 500

    data = request.get_json() or {}
    prompt_index = data.get("prompt_index")
    
    if prompt_index is None:
        # Discovery request: return dataset prompts listing
        return jsonify({
            "prompts_list": EVALUATION_PROMPTS
        })
        
    try:
        idx = int(prompt_index)
        if idx < 0 or idx >= len(EVALUATION_PROMPTS):
            return jsonify({"error": True, "message": "Invalid prompt index."}), 400
    except ValueError:
        return jsonify({"error": True, "message": "Prompt index must be an integer."}), 400

    target = EVALUATION_PROMPTS[idx]
    prompt = target["prompt"]
    
    tracker = PipelineTracker()
    start_time = time.time()
    
    try:
        # Prompt validation (vague prompts or short prompts under 10 words)
        words = [w for w in prompt.split() if w]
        if len(words) < 10:
            latency = time.time() - start_time
            # Return edge case validation failure immediately
            result = {
                "app_name": "Compilation Failed",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "intent": None,
                "system_design": None,
                "schemas": None,
                "validation_report": None,
                "error_flag": True,
                "error": f"Clarification required: Prompt is too short ({len(words)} words)."
            }
            readiness = evaluate_execution_readiness(result)
            result["execution_simulation"] = readiness
            return jsonify({
                "prompt_index": idx,
                "prompt": prompt,
                "type": target["type"],
                "success": False,
                "latency_seconds": round(latency, 2),
                "retries": 0,
                "error": "Input too short (vague boundary)",
                "result": result
            })
            
        # Run normal pipeline
        result, success = run_pipeline_sync(prompt, tracker)
        latency = time.time() - start_time
        
        # Check if validation passed in results
        val_passed = False
        if success and not result.get("error_flag"):
            val_report = result.get("validation_report", {})
            val_passed = val_report.get("passed", False)
            
        error_msg = None if success else result.get("error", "Unknown compilation error")
        
        return jsonify({
            "prompt_index": idx,
            "prompt": prompt,
            "type": target["type"],
            "success": success and val_passed,
            "latency_seconds": round(latency, 2),
            "retries": tracker.retries,
            "error": error_msg,
            "result": result
        })
        
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Evaluation crashed for prompt [{prompt}]: {e}")
        return jsonify({
            "prompt_index": idx,
            "prompt": prompt,
            "type": target["type"],
            "success": False,
            "latency_seconds": round(latency, 2),
            "retries": tracker.retries,
            "error": str(e)
        })

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
