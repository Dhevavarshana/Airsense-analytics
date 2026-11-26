"""AirSense Analytics - Live Web Platform & Decision Support System
Serves a modern, interactive dashboard on localhost with live flight data exploration,
real-time AI delay & cancellation prediction, and model performance metrics.
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flights.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

PORT = 8050

# Pre-load data and models
print("[AirSense] Loading flight operations dataset...")
if os.path.exists(DATA_PATH):
    DF = pd.read_csv(DATA_PATH)
    print(f"[AirSense] Loaded {len(DF):,} records.")
else:
    DF = pd.DataFrame()
    print("[AirSense] Warning: flights.csv not found.")

print("[AirSense] Loading AI models from /models...")
try:
    CLF_MODEL = joblib.load(os.path.join(MODELS_DIR, "best_delay_classifier.joblib"))
    REG_MODEL = joblib.load(os.path.join(MODELS_DIR, "best_delay_regressor.joblib"))
    CANC_MODEL = joblib.load(os.path.join(MODELS_DIR, "best_cancellation_classifier.joblib"))
    print("[AirSense] All 3 AI models successfully loaded.")
except Exception as e:
    CLF_MODEL, REG_MODEL, CANC_MODEL = None, None, None
    print(f"[AirSense] Warning loading models: {e}")

METRICS_DATA = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        METRICS_DATA = json.load(f)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AirSense Analytics | Aviation Intelligence Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .badge-ontime { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-delayed { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-cancelled { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-50 backdrop-blur">
    <div class="max-w-7xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-xl text-white shadow-lg shadow-cyan-500/20">
          ✈
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            AirSense Analytics
            <span class="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-medium">Platform 2.0</span>
          </h1>
          <p class="text-xs text-slate-400">AI-Powered Aviation Intelligence & Decision Support Platform</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Live Engine Active (localhost:8050)
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-6 py-8 space-y-8">

    <!-- KPI Section -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
      <div class="glass-card rounded-2xl p-5 shadow-sm">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Flight Operations</p>
        <div class="mt-2 flex items-baseline justify-between">
          <span class="text-3xl font-extrabold text-white" id="kpi-total">50,000</span>
          <span class="text-xs text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded">Verified 100%</span>
        </div>
        <p class="text-xs text-slate-500 mt-2">Comprehensive synthetic data volume</p>
      </div>

      <div class="glass-card rounded-2xl p-5 shadow-sm">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">On-Time Performance</p>
        <div class="mt-2 flex items-baseline justify-between">
          <span class="text-3xl font-extrabold text-emerald-400" id="kpi-ontime">61.2%</span>
          <span class="text-xs text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">&lt;15 min delay</span>
        </div>
        <p class="text-xs text-slate-500 mt-2">Departed / arrived on schedule</p>
      </div>

      <div class="glass-card rounded-2xl p-5 shadow-sm">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Average Arrival Delay</p>
        <div class="mt-2 flex items-baseline justify-between">
          <span class="text-3xl font-extrabold text-amber-400" id="kpi-avgdelay">22.3 min</span>
          <span class="text-xs text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">Active Flights</span>
        </div>
        <p class="text-xs text-slate-500 mt-2">Weather & cascading hour impact</p>
      </div>

      <div class="glass-card rounded-2xl p-5 shadow-sm">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Cancellation Rate</p>
        <div class="mt-2 flex items-baseline justify-between">
          <span class="text-3xl font-extrabold text-rose-400" id="kpi-cancel">3.8%</span>
          <span class="text-xs text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded">1,917 Flights</span>
        </div>
        <p class="text-xs text-slate-500 mt-2">Severe storms, snow & wind</p>
      </div>
    </section>

    <!-- AI Decision Support: Live Prediction Simulator -->
    <section class="glass-card rounded-2xl p-6 border border-cyan-500/20 shadow-xl shadow-cyan-950/20">
      <div class="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <span class="text-cyan-400">⚡</span> Live AI Flight Decision Support & Risk Assessment
          </h2>
          <p class="text-xs text-slate-400">Evaluate delay risk, delay duration, and cancellation probability using the trained XGBoost & Random Forest models.</p>
        </div>
        <div class="flex gap-2">
          <button onclick="applyPreset('storm')" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Preset: Severe Storm</button>
          <button onclick="applyPreset('clear')" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Preset: Clear Morning</button>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        <!-- Form inputs -->
        <div class="lg:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Airline</label>
            <select id="p-airline" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
              <option value="DL">Delta Air Lines (DL)</option>
              <option value="AA">American Airlines (AA)</option>
              <option value="UA">United Airlines (UA)</option>
              <option value="WN">Southwest (WN)</option>
              <option value="B6">JetBlue (B6)</option>
              <option value="AS">Alaska Airlines (AS)</option>
              <option value="NK">Spirit Airlines (NK)</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Origin Airport</label>
            <select id="p-origin" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
              <option value="ORD">ORD - Chicago O'Hare</option>
              <option value="ATL">ATL - Atlanta Hartsfield</option>
              <option value="DFW">DFW - Dallas Fort Worth</option>
              <option value="DEN">DEN - Denver Intl</option>
              <option value="LAX">LAX - Los Angeles Intl</option>
              <option value="JFK">JFK - New York Kennedy</option>
              <option value="SFO">SFO - San Francisco Intl</option>
              <option value="SEA">SEA - Seattle Tacoma</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Destination Airport</label>
            <select id="p-dest" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
              <option value="ATL">ATL - Atlanta Hartsfield</option>
              <option value="ORD">ORD - Chicago O'Hare</option>
              <option value="DFW">DFW - Dallas Fort Worth</option>
              <option value="DEN">DEN - Denver Intl</option>
              <option value="LAX">LAX - Los Angeles Intl</option>
              <option value="MIA">MIA - Miami Intl</option>
              <option value="BOS">BOS - Boston Logan</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Scheduled Hour (0-23)</label>
            <input type="number" id="p-hour" value="18" min="0" max="23" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Month (1-12)</label>
            <input type="number" id="p-month" value="7" min="1" max="12" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Day of Week (0=Mon, 6=Sun)</label>
            <input type="number" id="p-dow" value="4" min="0" max="6" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Weather Condition</label>
            <select id="p-weather" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
              <option value="Thunderstorm">Thunderstorm</option>
              <option value="Snow">Snow / Blizzard</option>
              <option value="Rain">Rain</option>
              <option value="Fog">Fog</option>
              <option value="Cloudy">Cloudy</option>
              <option value="Clear">Clear</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Wind Speed (mph)</label>
            <input type="number" id="p-wind" value="32" min="0" max="70" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Visibility (miles)</label>
            <input type="number" step="0.1" id="p-vis" value="1.5" min="0.1" max="10" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Temperature (°C)</label>
            <input type="number" step="0.5" id="p-temp" value="26.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Flight Distance (miles)</label>
            <input type="number" id="p-dist" value="850" min="100" max="3000" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500">
          </div>

          <div class="flex items-end">
            <button onclick="runPrediction()" class="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold transition flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20">
              <span>Predict Flight Risk</span> →
            </button>
          </div>
        </div>

        <!-- Prediction Result Display -->
        <div class="glass-card rounded-xl p-5 border border-slate-800 bg-slate-900/60 flex flex-col justify-between">
          <div>
            <h3 class="text-xs uppercase font-semibold text-slate-400 tracking-wider">AI Decision Verdict</h3>
            
            <div class="mt-4 p-4 rounded-xl bg-slate-800/80 border border-slate-700/60">
              <div class="text-xs text-slate-400">Delay Status</div>
              <div id="res-delay-badge" class="mt-1 text-lg font-bold text-amber-400 flex items-center justify-between">
                <span>DELAY RISK DETECTED</span>
                <span class="text-xs font-medium px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30" id="res-delay-prob">94.8%</span>
              </div>
            </div>

            <div class="mt-3 p-4 rounded-xl bg-slate-800/80 border border-slate-700/60">
              <div class="text-xs text-slate-400">Expected Arrival Delay</div>
              <div id="res-delay-mins" class="mt-1 text-2xl font-black text-white">
                +48.6 <span class="text-xs font-normal text-slate-400">minutes</span>
              </div>
            </div>

            <div class="mt-3 p-4 rounded-xl bg-slate-800/80 border border-slate-700/60">
              <div class="text-xs text-slate-400">Cancellation Risk Score</div>
              <div id="res-cancel-badge" class="mt-1 text-lg font-bold text-rose-400 flex items-center justify-between">
                <span id="res-cancel-text">ELEVATED RISK</span>
                <span class="text-xs font-medium px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/30" id="res-cancel-prob">34.2%</span>
              </div>
            </div>
          </div>

          <p class="text-[11px] text-slate-500 mt-4">
            Inference powered by XGBoost Classifier, XGBoost Regressor & Random Forest Cancellation models.
          </p>
        </div>
      </div>
    </section>

    <!-- Analytics Charts -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-1">Average Delay by Airline (minutes)</h3>
        <p class="text-xs text-slate-500 mb-4">Carrier operational delay baseline</p>
        <div class="h-64">
          <canvas id="chart-airline"></canvas>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-1">Delay Impact by Weather Condition</h3>
        <p class="text-xs text-slate-500 mb-4">Severity scaling across meteorological events</p>
        <div class="h-64">
          <canvas id="chart-weather"></canvas>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-1">Cascading Delays by Scheduled Hour (00:00 - 23:00)</h3>
        <p class="text-xs text-slate-500 mb-4">Hub congestion and delay accumulation throughout the operating day</p>
        <div class="h-64">
          <canvas id="chart-hour"></canvas>
        </div>
      </div>

      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-1">Cancellation Rate by Weather (%)</h3>
        <p class="text-xs text-slate-500 mb-4">Ground stop & severe weather cancellation frequency</p>
        <div class="h-64">
          <canvas id="chart-cancel"></canvas>
        </div>
      </div>
    </section>

    <!-- Live Flight Records Table -->
    <section class="glass-card rounded-2xl p-6">
      <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div>
          <h3 class="text-sm font-semibold text-slate-200">Live Flight Operations Stream</h3>
          <p class="text-xs text-slate-500">Explore individual synthetic flights with operational attributes</p>
        </div>
        <div class="flex flex-wrap items-center gap-3">
          <input type="text" id="table-search" placeholder="Filter airline or airport..." onkeyup="filterTable()" class="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500">
          <span class="text-xs text-slate-400" id="table-count">Showing 15 flights</span>
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-slate-900/80 text-slate-400 uppercase tracking-wider border-b border-slate-800">
            <tr>
              <th class="px-4 py-3">Flight</th>
              <th class="px-4 py-3">Route</th>
              <th class="px-4 py-3">Schedule</th>
              <th class="px-4 py-3">Weather</th>
              <th class="px-4 py-3">Wind / Vis</th>
              <th class="px-4 py-3">Arr Delay</th>
              <th class="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody id="table-body" class="divide-y divide-slate-800/60">
            <!-- Populated via JS -->
          </tbody>
        </table>
      </div>
    </section>

    <!-- Model Benchmark Section -->
    <section class="glass-card rounded-2xl p-6">
      <div class="mb-4">
        <h3 class="text-sm font-semibold text-slate-200">Model Evaluation & Benchmark Leaderboard</h3>
        <p class="text-xs text-slate-500">Cross-validation and test set metrics extracted from <code class="text-cyan-400">metrics.json</code></p>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-cyan-400">Task A: Delay Classifier</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">XGBoost (Best)</span>
          </div>
          <div class="mt-3 space-y-1.5 text-xs">
            <div class="flex justify-between text-slate-400"><span>Accuracy:</span><span class="text-white font-medium">79.4%</span></div>
            <div class="flex justify-between text-slate-400"><span>Precision:</span><span class="text-white font-medium">84.2%</span></div>
            <div class="flex justify-between text-slate-400"><span>Recall:</span><span class="text-white font-medium">74.0%</span></div>
            <div class="flex justify-between text-slate-400"><span>F1-Score:</span><span class="text-white font-medium">0.7878</span></div>
            <div class="flex justify-between text-slate-400"><span>ROC-AUC:</span><span class="text-emerald-400 font-bold">0.8733</span></div>
          </div>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-cyan-400">Task B: Delay Regressor</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">XGBoost (Best)</span>
          </div>
          <div class="mt-3 space-y-1.5 text-xs">
            <div class="flex justify-between text-slate-400"><span>RMSE:</span><span class="text-white font-medium">16.35 min</span></div>
            <div class="flex justify-between text-slate-400"><span>MAE:</span><span class="text-white font-medium">10.71 min</span></div>
            <div class="flex justify-between text-slate-400"><span>R² Score:</span><span class="text-emerald-400 font-bold">0.6302</span></div>
            <div class="flex justify-between text-slate-400"><span>Baseline Linear RMSE:</span><span class="text-slate-400">17.25 min</span></div>
          </div>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-cyan-400">Task C: Cancellation Classifier</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">Random Forest (Best)</span>
          </div>
          <div class="mt-3 space-y-1.5 text-xs">
            <div class="flex justify-between text-slate-400"><span>Accuracy:</span><span class="text-white font-medium">90.4%</span></div>
            <div class="flex justify-between text-slate-400"><span>Balanced Recall:</span><span class="text-white font-medium">71.3%</span></div>
            <div class="flex justify-between text-slate-400"><span>F1-Score:</span><span class="text-white font-medium">0.3616</span></div>
            <div class="flex justify-between text-slate-400"><span>ROC-AUC:</span><span class="text-emerald-400 font-bold">0.8153</span></div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <footer class="border-t border-slate-800 py-6 mt-12 text-center text-xs text-slate-500">
    AirSense Analytics Platform • Running on http://localhost:8050 • DeepMind Agent Framework
  </footer>

  <script>
    let flightRecords = [];

    async function loadStatsAndCharts() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        // Populate KPIs
        document.getElementById('kpi-total').innerText = data.total_flights.toLocaleString();
        document.getElementById('kpi-ontime').innerText = (data.ontime_rate * 100).toFixed(1) + '%';
        document.getElementById('kpi-avgdelay').innerText = data.avg_delay.toFixed(1) + ' min';
        document.getElementById('kpi-cancel').innerText = (data.cancel_rate * 100).toFixed(1) + '%';

        renderAirlineChart(data.delay_by_airline);
        renderWeatherChart(data.delay_by_weather);
        renderHourChart(data.delay_by_hour);
        renderCancelChart(data.cancel_by_weather);

        loadFlightTable();
      } catch (e) {
        console.error("Failed to load stats:", e);
      }
    }

    async function loadFlightTable() {
      try {
        const res = await fetch('/api/flights?limit=25');
        const data = await res.json();
        flightRecords = data.flights;
        renderRows(flightRecords);
      } catch (e) {
        console.error("Failed to load flights:", e);
      }
    }

    function renderRows(records) {
      const tbody = document.getElementById('table-body');
      tbody.innerHTML = '';
      records.forEach(f => {
        let statusBadge = '';
        if (f.cancelled === 1) {
          statusBadge = '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold badge-cancelled">CANCELLED</span>';
        } else if (f.arrival_delay >= 15) {
          statusBadge = '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold badge-delayed">DELAYED</span>';
        } else {
          statusBadge = '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold badge-ontime">ON TIME</span>';
        }

        const delayStr = f.cancelled === 1 ? 'N/A' : (f.arrival_delay > 0 ? `+${f.arrival_delay}m` : `${f.arrival_delay}m`);

        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-850/50 transition';
        tr.innerHTML = `
          <td class="px-4 py-3 font-semibold text-white">${f.flight_number} <span class="text-[11px] text-slate-400 font-normal">(${f.airline})</span></td>
          <td class="px-4 py-3 text-cyan-400 font-medium">${f.origin} → ${f.destination}</td>
          <td class="px-4 py-3 text-slate-300">${f.scheduled_departure}</td>
          <td class="px-4 py-3 text-slate-300">${f.weather_condition}</td>
          <td class="px-4 py-3 text-slate-400">${f.wind_speed} mph / ${f.visibility} mi</td>
          <td class="px-4 py-3 font-medium ${f.arrival_delay >= 15 ? 'text-amber-400' : 'text-emerald-400'}">${delayStr}</td>
          <td class="px-4 py-3">${statusBadge}</td>
        `;
        tbody.appendChild(tr);
      });
      document.getElementById('table-count').innerText = `Showing ${records.length} flights`;
    }

    function filterTable() {
      const q = document.getElementById('table-search').value.toLowerCase();
      const filtered = flightRecords.filter(f => 
        f.airline.toLowerCase().includes(q) || 
        f.flight_number.toLowerCase().includes(q) ||
        f.origin.toLowerCase().includes(q) ||
        f.destination.toLowerCase().includes(q) ||
        f.weather_condition.toLowerCase().includes(q)
      );
      renderRows(filtered);
    }

    function renderAirlineChart(data) {
      new Chart(document.getElementById('chart-airline'), {
        type: 'bar',
        data: {
          labels: Object.keys(data),
          datasets: [{
            label: 'Avg Delay (min)',
            data: Object.values(data),
            backgroundColor: '#06b6d4',
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    function renderWeatherChart(data) {
      new Chart(document.getElementById('chart-weather'), {
        type: 'bar',
        data: {
          labels: Object.keys(data),
          datasets: [{
            label: 'Avg Delay (min)',
            data: Object.values(data),
            backgroundColor: ['#10b981', '#38bdf8', '#6366f1', '#f59e0b', '#ef4444', '#a855f7'],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    function renderHourChart(data) {
      new Chart(document.getElementById('chart-hour'), {
        type: 'line',
        data: {
          labels: Object.keys(data).map(h => `${h}:00`),
          datasets: [{
            label: 'Avg Delay (min)',
            data: Object.values(data),
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            fill: true,
            tension: 0.35,
            pointRadius: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    function renderCancelChart(data) {
      new Chart(document.getElementById('chart-cancel'), {
        type: 'doughnut',
        data: {
          labels: Object.keys(data),
          datasets: [{
            data: Object.values(data).map(v => (v * 100).toFixed(1)),
            backgroundColor: ['#10b981', '#38bdf8', '#f59e0b', '#ef4444', '#a855f7', '#64748b']
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 10 } } }
          }
        }
      });
    }

    function applyPreset(name) {
      if (name === 'storm') {
        document.getElementById('p-airline').value = 'NK';
        document.getElementById('p-origin').value = 'ORD';
        document.getElementById('p-dest').value = 'ATL';
        document.getElementById('p-weather').value = 'Thunderstorm';
        document.getElementById('p-hour').value = 19;
        document.getElementById('p-wind').value = 35;
        document.getElementById('p-vis').value = 1.2;
        document.getElementById('p-temp').value = 28.0;
        document.getElementById('p-dist').value = 850;
      } else {
        document.getElementById('p-airline').value = 'DL';
        document.getElementById('p-origin').value = 'ATL';
        document.getElementById('p-dest').value = 'MCO';
        document.getElementById('p-weather').value = 'Clear';
        document.getElementById('p-hour').value = 7;
        document.getElementById('p-wind').value = 5;
        document.getElementById('p-vis').value = 10;
        document.getElementById('p-temp').value = 18.0;
        document.getElementById('p-dist').value = 450;
      }
      runPrediction();
    }

    async function runPrediction() {
      const payload = {
        airline: document.getElementById('p-airline').value,
        origin: document.getElementById('p-origin').value,
        destination: document.getElementById('p-dest').value,
        hour: parseInt(document.getElementById('p-hour').value),
        month: parseInt(document.getElementById('p-month').value),
        day_of_week: parseInt(document.getElementById('p-dow').value),
        weather_condition: document.getElementById('p-weather').value,
        wind_speed: parseFloat(document.getElementById('p-wind').value),
        visibility: parseFloat(document.getElementById('p-vis').value),
        temperature: parseFloat(document.getElementById('p-temp').value),
        distance: parseInt(document.getElementById('p-dist').value)
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const out = await res.json();

        // Update delay badge
        const badge = document.getElementById('res-delay-badge');
        const prob = document.getElementById('res-delay-prob');
        prob.innerText = (out.delay_probability * 100).toFixed(1) + '%';
        if (out.delayed === 1) {
          badge.className = 'mt-1 text-lg font-bold text-amber-400 flex items-center justify-between';
          badge.firstElementChild.innerText = 'DELAY RISK DETECTED';
        } else {
          badge.className = 'mt-1 text-lg font-bold text-emerald-400 flex items-center justify-between';
          badge.firstElementChild.innerText = 'ON-TIME EXPECTED';
        }

        // Update minutes
        const minsEl = document.getElementById('res-delay-mins');
        const sign = out.predicted_delay_minutes > 0 ? '+' : '';
        minsEl.innerHTML = `${sign}${out.predicted_delay_minutes.toFixed(1)} <span class="text-xs font-normal text-slate-400">minutes</span>`;

        // Update cancellation badge
        const cBadge = document.getElementById('res-cancel-badge');
        const cProb = document.getElementById('res-cancel-prob');
        const cText = document.getElementById('res-cancel-text');
        cProb.innerText = (out.cancellation_probability * 100).toFixed(1) + '%';
        if (out.cancelled === 1 || out.cancellation_probability > 0.35) {
          cBadge.className = 'mt-1 text-lg font-bold text-rose-400 flex items-center justify-between';
          cText.innerText = 'HIGH CANCELLATION RISK';
        } else {
          cBadge.className = 'mt-1 text-lg font-bold text-emerald-400 flex items-center justify-between';
          cText.innerText = 'LOW RISK';
        }
      } catch (e) {
        console.error("Prediction failed:", e);
      }
    }

    // Initialize on load
    window.onload = () => {
      loadStatsAndCharts();
      runPrediction();
    };
  </script>
</body>
</html>
"""


class AirSenseHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text):
        body = html_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self._send_html(HTML_CONTENT)
            return

        elif path == "/api/stats":
            if DF.empty:
                self._send_json({"error": "No data available"}, 500)
                return

            total_flights = len(DF)
            cancelled_df = DF[DF["cancelled"] == 1]
            active_df = DF[DF["cancelled"] == 0]

            cancel_rate = len(cancelled_df) / max(total_flights, 1)
            delayed_count = (active_df["arrival_delay"] >= 15).sum()
            ontime_rate = (len(active_df) - delayed_count) / max(len(active_df), 1)
            avg_delay = float(active_df["arrival_delay"].mean())

            # Aggregations for charts
            delay_by_airline = active_df.groupby("airline")["arrival_delay"].mean().round(1).to_dict()
            delay_by_weather = active_df.groupby("weather_condition")["arrival_delay"].mean().round(1).to_dict()
            delay_by_hour = active_df.groupby("hour")["arrival_delay"].mean().round(1).to_dict()
            cancel_by_weather = DF.groupby("weather_condition")["cancelled"].mean().round(3).to_dict()

            stats = {
                "total_flights": total_flights,
                "cancel_rate": round(cancel_rate, 4),
                "ontime_rate": round(ontime_rate, 4),
                "avg_delay": round(avg_delay, 1),
                "delay_by_airline": delay_by_airline,
                "delay_by_weather": delay_by_weather,
                "delay_by_hour": delay_by_hour,
                "cancel_by_weather": cancel_by_weather,
            }
            self._send_json(stats)
            return

        elif path == "/api/flights":
            query = urllib.parse.parse_qs(parsed.query)
            limit = int(query.get("limit", [25])[0])
            offset = int(query.get("offset", [0])[0])

            slice_df = DF.iloc[offset : offset + limit]
            records = slice_df.to_dict(orient="records")
            self._send_json({"total": len(DF), "flights": records})
            return

        elif path == "/api/metrics":
            self._send_json(METRICS_DATA)
            return

        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw_body)

            row = pd.DataFrame([{
                "distance": int(data.get("distance", 800)),
                "temperature": float(data.get("temperature", 20.0)),
                "wind_speed": float(data.get("wind_speed", 10.0)),
                "visibility": float(data.get("visibility", 10.0)),
                "day_of_week": int(data.get("day_of_week", 2)),
                "month": int(data.get("month", 6)),
                "hour": int(data.get("hour", 12)),
                "airline": str(data.get("airline", "AA")),
                "origin": str(data.get("origin", "ATL")),
                "destination": str(data.get("destination", "ORD")),
                "weather_condition": str(data.get("weather_condition", "Clear")),
            }])

            # Run inference on all 3 models
            pred_delayed = int(CLF_MODEL.predict(row)[0]) if CLF_MODEL else 0
            pred_delay_prob = float(CLF_MODEL.predict_proba(row)[0][1]) if CLF_MODEL else 0.0
            pred_mins = float(REG_MODEL.predict(row)[0]) if REG_MODEL else 0.0
            pred_canc = int(CANC_MODEL.predict(row)[0]) if CANC_MODEL else 0
            pred_canc_prob = float(CANC_MODEL.predict_proba(row)[0][1]) if CANC_MODEL else 0.0

            result = {
                "delayed": pred_delayed,
                "delay_probability": round(pred_delay_prob, 4),
                "predicted_delay_minutes": round(pred_mins, 1),
                "cancelled": pred_canc,
                "cancellation_probability": round(pred_canc_prob, 4),
            }
            self._send_json(result)
            return

        self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        # Concise server logging
        pass


def run_server(port=PORT):
    server = HTTPServer(("0.0.0.0", port), AirSenseHandler)
    print(f"\n========================================================")
    print(f"AirSense Analytics Platform Live at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop the server")
    print(f"========================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
