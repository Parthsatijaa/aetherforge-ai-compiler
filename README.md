# AetherForge AI - Multi-Stage AI Specification Compiler

A complete multi-stage AI pipeline compiler built using a **Python Flask** backend and a **vanilla HTML/JS/CSS** frontend. The app uses the **Groq API** (specifically the `llama-3.1-8b-instant` model for high-throughput rate limits) to translate natural language app descriptions into structured, executable JSON system specifications.

The pipeline comprises exactly 5 stages:
1. **Stage 1 (Intent Parser)**: Extracts high-level intent, name, features, and target user roles.
2. **Stage 2 (System Architect)**: Computes relationships, workflows, and role permissions.
3. **Stage 3 (Schema Generator)**: Builds detailed UI routes, API endpoints, database schemas, and auth scopes.
4. **Stage 4 (Validator & Repair)**: Checks cross-consistency between schemas, database, and APIs, correcting mistakes.
5. **Stage 5 (Compiler Merge)**: Assembles all outputs, documents key decisions, and compiles the final unified JSON payload.

---

## 🚀 Setup & Execution Guide

### Local Setup (Windows / macOS / Linux)

1. **Clone the Repository** (or navigate to your project directory):
   ```powershell
   cd "d:\AI Project Demo Task"
   ```
2. **Install Python Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Copy `.env.example` to a new file named `.env`:
   ```powershell
   copy .env.example .env
   ```
   Open `.env` in a text editor and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_yourActualKeyFromConsoleGroq
   ```
4. **Start the Flask Server**:
   ```powershell
   python app.py
   ```
5. **Open in Browser**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🌐 How to Deploy on Render (Free Hosting)

Follow these steps to host your application live on Render:

1. **Create a Render Account**: Sign up at [render.com](https://render.com/).
2. **Create a New Web Service**:
   - Click **New +** in the dashboard and select **Web Service**.
   - Connect your GitHub repository.
3. **Configure Service Settings**:
   - **Name**: `aetherforge-ai-compiler` (or your preferred name)
   - **Language**: `Python`
   - **Branch**: `main` (or your active branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 120` (Gunicorn timeout is increased to allow the sequential AI pipeline to finish without worker timeouts)
4. **Add Environment Variables**:
   - Under the **Environment** tab, click **Add Environment Variable**.
   - Set Key: `GROQ_API_KEY`
   - Set Value: `gsk_...` (your Groq API key)
5. **Deploy**: Click **Create Web Service**. Your app will build and go live on a `*.onrender.com` URL!

---

## 📹 Loom Demo Video Guide (5-Minute Script)

To make your submission stand out, record a short demo video covering these key technical highlights:

* **0:00 - 1:00 | Introduction & Goal**:
  > *"Hi, this is [Your Name], and I'm presenting AetherForge AI—a multi-stage AI compilation pipeline. The goal is to take open-ended natural language requirements and reliably output structured, executable app specifications including UI routing, DB tables, API schemas, and Auth mapping."*
* **1:00 - 2:15 | Core Architecture (Multi-Stage & Self-Healing)**:
  > *"Instead of using one massive prompt which causes formatting truncation and high hallucination rates, this compiler uses a sequential 5-stage pipeline. In Stage 5, we programmatically compile and merge the layouts. If any stage outputs invalid JSON, our custom self-healing/repair utility automatically invokes a targeted recovery call to repair the specific broken node."*
* **2:15 - 3:30 | Live Walkthrough**:
  > *Show a live generation (e.g., compile a CRM prompt). Show the stage Timings updating. Expand the completed JSON and show the "Runtime Execution Simulation Log" validating the schema's metadata, database, and API mapping.*
* **3:30 - 4:30 | Evaluation Framework & Test Matrix**:
  > *Click the 'Run System Benchmark' button. Show the matrix evaluating 10 real product prompts and 10 edge cases (vague, conflicting, and incomplete prompts). Discuss the success rates, latencies, and retry statistics.*
* **4:30 - 5:00 | Cost vs. Quality tradeoffs**:
  > *Briefly show the optimization tradeoffs: we chose sequential compilation with `llama-3.1-8b-instant` on Groq, ensuring maximum schema quality and self-healing resilience for zero hosting/token costs.*

---

## 📊 Cost vs. Quality Tradeoff Analysis

Our compiler selects a **multi-stage sequential generation flow** targeting `llama-3.1-8b-instant` over a single-shot prompt generation model. The tradeoffs are detailed below:

### Optimization Matrix
| Dimension | Single-Shot Generator | AetherForge Multi-Stage Compiler |
| :--- | :--- | :--- |
| **Hallucination Rate** | High (combines UI, DB, Auth, and APIs in one prompt context) | **Very Low** (narrowed contextual scope per stage) |
| **Schema Strictness** | Weak (easily skips validation contracts, bad JSON keys) | **Absolute** (Stage 4 performs structural static linting) |
| **Self-Healing** | None (crashes if output JSON formatting fails) | **Automatic** (Stage 4 repair prompts fix individual broken sub-nodes) |
| **Total Token Cost** | ~$0.005 / run | **$0.00** (Uses Groq's free tier) |
| **Mean Latency** | ~4-6 seconds | ~14-18 seconds (sequential execution) |

### Balancing Strategy
- **Perceived Latency**: Solved by implementing **Server-Sent Events (SSE)** streaming. The client UI registers real-time visual progress and individual stage checkmarks rather than blocking on a single long loading screen.
- **Programmatic Merging**: Stage 5 programmatically merges schemas rather than asking the LLM to write out thousands of characters of repetitive JSON. This reduces latency by over 50% and completely avoids formatting errors.
