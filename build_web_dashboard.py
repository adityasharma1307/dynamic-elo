"""
Build a single self-contained HTML dashboard (docs/index.html) from the
report files produced by baseline.py and dynamic.py, for showcasing on
GitHub Pages / Vercel / Netlify -- no backend, no build step, one file.

Run baseline.py then dynamic.py first to refresh the source reports, then
run this script to refresh the web dashboard.
"""
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS  = os.path.join(BASE_DIR, "data", "reports")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

BASELINE_LOG    = os.path.join(REPORTS, "2024_GS_Baseline_Match_Log.xlsx")
DYNAMIC_LOG     = os.path.join(REPORTS, "2024_GS_Dynamic_ELO_Match_Log.xlsx")
DYNAMIC_COMPARE = os.path.join(REPORTS, "Dynamic_ELO_Split_Comparison.xlsx")
OUT_PATH        = os.path.join(DOCS_DIR, "index.html")


def main():
    if not (os.path.exists(BASELINE_LOG) and os.path.exists(DYNAMIC_LOG) and os.path.exists(DYNAMIC_COMPARE)):
        raise FileNotFoundError(
            "Missing source reports. Run `python baseline.py` then `python dynamic.py` "
            "first to generate the files this dashboard is built from."
        )

    base_summary = pd.read_excel(BASELINE_LOG, sheet_name="Summary")
    dyn_summary  = pd.read_excel(DYNAMIC_LOG, sheet_name="Summary")
    dyn_matches  = pd.read_excel(DYNAMIC_LOG, sheet_name="All Matches")
    omega        = pd.read_excel(DYNAMIC_COMPARE, sheet_name="Omega Feature Search")
    fi           = pd.read_excel(DYNAMIC_COMPARE, sheet_name="Feature Importance")

    base_ov = base_summary.loc[base_summary["Tournament"] == "OVERALL"].iloc[0]
    dyn_ov  = dyn_summary.loc[dyn_summary["Tournament"] == "OVERALL"].iloc[0]

    slams = [s for s in dyn_summary["Tournament"] if s != "OVERALL"]
    base_by_slam = base_summary.set_index("Tournament")
    dyn_by_slam  = dyn_summary.set_index("Tournament")

    data = {
        "kpis": {
            "baseline_acc": round(float(base_ov["Accuracy (%)"]), 1),
            "dynamic_acc": round(float(dyn_ov["Accuracy (%)"]), 1),
            "delta_pp": round(float(dyn_ov["Accuracy (%)"]) - float(base_ov["Accuracy (%)"]), 1),
            "matches": int(dyn_ov["Matches"]),
            "baseline_auc": None,
            "dynamic_auc": None,
        },
        "slams": slams,
        "baseline_acc_by_slam": [round(float(base_by_slam.loc[s, "Accuracy (%)"]), 1) for s in slams],
        "dynamic_acc_by_slam": [round(float(dyn_by_slam.loc[s, "Accuracy (%)"]), 1) for s in slams],
        "omega": {
            "labels": omega["Omega Variant"].tolist(),
            "accuracy": [round(float(v), 2) for v in omega["Test Accuracy (%)"]],
        },
        "feature_importance": {
            "labels": fi.sort_values("Gain (%)", ascending=False)["Feature"].tolist(),
            "gain": [round(float(v), 2) for v in fi.sort_values("Gain (%)", ascending=False)["Gain (%)"]],
        },
        "matches": json.loads(
            dyn_matches[
                ["Tournament", "Round", "Surface", "Winner", "Loser",
                 "Win Probability", "Confidence", "Prediction", "Correct?", "Upset?"]
            ].to_json(orient="records")
        ),
    }

    # AUC (not in Summary sheets -- pull the known headline values written to README)
    data["kpis"]["baseline_auc"] = 0.8157
    data["kpis"]["dynamic_auc"] = 0.8164

    os.makedirs(DOCS_DIR, exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Web dashboard saved: {OUT_PATH}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynamic ELO — Tennis Grand Slam Prediction Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    --navy:#111827; --slate:#374151; --muted:#6b7280; --line:#e5e7eb;
    --blue:#2563eb; --green:#16a34a; --amber:#d97706; --bg:#f9fafb;
    --card:#ffffff;
  }
  * { box-sizing: border-box; }
  body {
    margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--navy);
  }
  header { padding: 32px 24px 16px; max-width: 1200px; margin: 0 auto; }
  header h1 { margin: 0 0 4px; font-size: 26px; }
  header p { margin: 0; color: var(--muted); font-size: 14px; }
  header .badges { margin-top: 10px; }
  header .badges span {
    display:inline-block; font-size:12px; padding:3px 10px; border-radius:999px;
    background:#eef2ff; color:#4338ca; margin-right:8px;
  }
  main { max-width: 1200px; margin: 0 auto; padding: 8px 24px 48px; }
  .kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap:16px; margin: 16px 0 28px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }
  .kpi .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .kpi .value { font-size:30px; font-weight:700; margin-top:4px; }
  .kpi.blue .value { color: var(--blue); }
  .kpi.green .value { color: var(--green); }
  .kpi.amber .value { color: var(--amber); }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }
  .card h2 { font-size:15px; margin:0 0 12px; color:var(--slate); }
  .card canvas { max-height: 320px; }
  .full { grid-column: 1 / -1; }
  table { width:100%; border-collapse: collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  th { position: sticky; top:0; background:#f3f4f6; color:var(--slate); font-weight:600; cursor:pointer; user-select:none; }
  tr:hover td { background:#fafafa; }
  .table-wrap { max-height: 480px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
  .pill { padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
  .pill.yes { background:#dcfce7; color:#166534; }
  .pill.no { background:#fee2e2; color:#991b1b; }
  .pill.upset { background:#fef9c3; color:#854d0e; }
  .controls { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
  .controls input, .controls select {
    padding:7px 10px; border:1px solid var(--line); border-radius:8px; font-size:13px;
  }
  footer { text-align:center; color:var(--muted); font-size:12px; padding: 20px 0 40px; }
  footer a { color: var(--blue); }
</style>
</head>
<body>
<header>
  <h1>Dynamic ELO — Tennis Grand Slam Prediction</h1>
  <p>Baseline vs. Dynamic ELO · 2024 Grand Slam season · XGBoost per-surface ensemble</p>
  <div class="badges">
    <span>xgboost</span><span>tennis analytics</span><span>elo rating</span><span>data science</span>
  </div>
</header>
<main>
  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="card">
      <h2>Per-Slam Test Accuracy — Baseline vs. Dynamic ELO</h2>
      <canvas id="slamChart"></canvas>
    </div>
    <div class="card">
      <h2>Ω Combination Search — Test Accuracy by Variant</h2>
      <canvas id="omegaChart"></canvas>
    </div>
    <div class="card full">
      <h2>XGBoost Feature Importance (Dynamic ELO Global Model)</h2>
      <canvas id="fiChart"></canvas>
    </div>
    <div class="card full">
      <h2>2024 Match-by-Match Predictions (Dynamic ELO)</h2>
      <div class="controls">
        <input id="search" type="text" placeholder="Search player, round, tournament...">
        <select id="slamFilter"><option value="">All Slams</option></select>
        <select id="correctFilter">
          <option value="">All Predictions</option>
          <option value="YES">Correct only</option>
          <option value="NO">Incorrect only</option>
        </select>
      </div>
      <div class="table-wrap">
        <table id="matchTable">
          <thead>
            <tr>
              <th data-key="Tournament">Tournament</th>
              <th data-key="Round">Round</th>
              <th data-key="Surface">Surface</th>
              <th data-key="Winner">Winner</th>
              <th data-key="Loser">Loser</th>
              <th data-key="Prediction">Prediction</th>
              <th data-key="Win Probability">Win Prob.</th>
              <th data-key="Correct?">Correct?</th>
              <th data-key="Upset?">Upset?</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<footer>
  Generated by <code>build_web_dashboard.py</code> from this repo's own report files &middot;
  <a href="https://github.com/JeffSackmann/tennis_atp" target="_blank">Data: Jeff Sackmann's tennis_atp</a> &amp;
  <a href="https://github.com/JeffSackmann/tennis_MatchChartingProject" target="_blank">MatchChartingProject</a>
</footer>

<script>
const DATA = __DATA_JSON__;

function fmtPct(x) { return x.toFixed(1) + '%'; }

function renderKPIs() {
  const el = document.getElementById('kpis');
  const k = DATA.kpis;
  const items = [
    { label: 'Baseline Accuracy', value: fmtPct(k.baseline_acc), cls: 'blue' },
    { label: 'Dynamic ELO Accuracy', value: fmtPct(k.dynamic_acc), cls: 'green' },
    { label: 'Improvement', value: (k.delta_pp >= 0 ? '+' : '') + k.delta_pp.toFixed(1) + 'pp', cls: 'amber' },
    { label: 'Matches Evaluated', value: k.matches, cls: '' },
    { label: 'Baseline AUC', value: k.baseline_auc.toFixed(4), cls: 'blue' },
    { label: 'Dynamic ELO AUC', value: k.dynamic_auc.toFixed(4), cls: 'green' },
  ];
  el.innerHTML = items.map(i => `
    <div class="kpi ${i.cls}">
      <div class="label">${i.label}</div>
      <div class="value">${i.value}</div>
    </div>`).join('');
}

function renderCharts() {
  new Chart(document.getElementById('slamChart'), {
    type: 'bar',
    data: {
      labels: DATA.slams,
      datasets: [
        { label: 'Baseline', data: DATA.baseline_acc_by_slam, backgroundColor: '#93c5fd' },
        { label: 'Dynamic ELO', data: DATA.dynamic_acc_by_slam, backgroundColor: '#16a34a' },
      ]
    },
    options: { responsive:true, plugins:{legend:{position:'bottom'}}, scales:{y:{beginAtZero:true, max:100, title:{display:true,text:'Accuracy (%)'}}} }
  });

  new Chart(document.getElementById('omegaChart'), {
    type: 'bar',
    data: {
      labels: DATA.omega.labels,
      datasets: [{ label: 'Test Accuracy (%)', data: DATA.omega.accuracy, backgroundColor: '#2563eb' }]
    },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Accuracy (%)'}}} }
  });

  new Chart(document.getElementById('fiChart'), {
    type: 'bar',
    data: {
      labels: DATA.feature_importance.labels,
      datasets: [{ label: 'Gain (%)', data: DATA.feature_importance.gain, backgroundColor: '#d97706' }]
    },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Gain (%)'}}} }
  });
}

let sortKey = null, sortDir = 1;
function renderTable() {
  const tbody = document.querySelector('#matchTable tbody');
  const search = document.getElementById('search').value.toLowerCase();
  const slam = document.getElementById('slamFilter').value;
  const correct = document.getElementById('correctFilter').value;

  let rows = DATA.matches.filter(r => {
    if (slam && r.Tournament !== slam) return false;
    if (correct && r['Correct?'] !== correct) return false;
    if (search) {
      const hay = `${r.Tournament} ${r.Round} ${r.Surface} ${r.Winner} ${r.Loser} ${r.Prediction}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });

  if (sortKey) {
    rows = rows.slice().sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === 'number') return (av - bv) * sortDir;
      return String(av).localeCompare(String(bv)) * sortDir;
    });
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.Tournament}</td>
      <td>${r.Round}</td>
      <td>${r.Surface}</td>
      <td>${r.Winner}</td>
      <td>${r.Loser}</td>
      <td>${r.Prediction}</td>
      <td>${(r['Win Probability']*100).toFixed(1)}%</td>
      <td><span class="pill ${r['Correct?']==='YES'?'yes':'no'}">${r['Correct?']}</span></td>
      <td>${r['Upset?']==='YES' ? '<span class="pill upset">YES</span>' : '-'}</td>
    </tr>`).join('');
}

function populateFilters() {
  const sel = document.getElementById('slamFilter');
  DATA.slams.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderKPIs();
  renderCharts();
  populateFilters();
  renderTable();
  document.getElementById('search').addEventListener('input', renderTable);
  document.getElementById('slamFilter').addEventListener('change', renderTable);
  document.getElementById('correctFilter').addEventListener('change', renderTable);
  document.querySelectorAll('#matchTable th').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      sortDir = (sortKey === key) ? -sortDir : 1;
      sortKey = key;
      renderTable();
    });
  });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
