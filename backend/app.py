#!/usr/bin/env python3
"""
AgroSpray backend - receives stops from the robot, stores them in SQLite,
serves the farmer dashboard.

Run:   python3 backend/app.py            (http://localhost:8000)
Env:   AGROSPRAY_KEY   shared secret the robot sends as X-API-Key (change it!)
       AGROSPRAY_DB    path of the SQLite file (default backend/agrospray.db)
       PORT            default 8000
"""
import csv
import io
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("AGROSPRAY_DB", BASE / "agrospray.db"))
UPLOADS = BASE / "uploads"
DASHBOARD = BASE.parent / "dashboard"
API_KEY = os.environ.get("AGROSPRAY_KEY", "agrospray-dev-key")
STATES = {"HEALTHY", "MILD", "MODERATE", "SEVERE", "DEAD", "NOLEAF"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024      # 5 MB per upload
UPLOADS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- database
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS stops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                robot_id TEXT, ts TEXT, source TEXT, label TEXT, conf REAL,
                damage_pct REAL, state TEXT, dose_ml_l REAL,
                spray_ms INTEGER, max_spray_ms INTEGER, image TEXT, created REAL
            )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_stops_created ON stops(created)")


def crop_of(label):
    if not label:
        return "unknown"
    for suffix in ("_healthy", "_diseased"):
        if label.lower().endswith(suffix):
            return label[: -len(suffix)]
    return label


def row_to_dict(r):
    d = dict(r)
    d["image_url"] = f"/uploads/{d['image']}" if d.get("image") else None
    d["crop"] = crop_of(d.get("label"))
    return d


# --------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.post("/api/stops")
def add_stop():
    if request.headers.get("X-API-Key") != API_KEY:
        abort(401)
    try:
        data = json.loads(request.form.get("data", "{}"))
        state = str(data["state"]).upper()
        if state not in STATES:
            raise ValueError("bad state")
        rec = (
            str(data.get("robot_id", "robot-1"))[:40],
            str(data.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S")),
            str(data.get("source", ""))[:120],
            (str(data["label"])[:80] if data.get("label") else None),
            float(data.get("conf", 0)),
            float(data["damage_pct"]),
            state,
            float(data.get("dose_ml_l", 0)),
            int(data.get("spray_ms", 0)),
            int(data.get("max_spray_ms", 0)),
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        abort(400, "invalid stop payload")

    image_name = None
    f = request.files.get("image")
    if f:
        blob = f.read()
        if blob[:2] != b"\xff\xd8":                      # JPEG magic bytes only
            abort(400, "image must be a JPEG")
        image_name = f"{uuid.uuid4().hex}.jpg"
        (UPLOADS / image_name).write_bytes(blob)

    with db() as con:
        cur = con.execute(
            "INSERT INTO stops (robot_id, ts, source, label, conf, damage_pct, state, dose_ml_l,"
            " spray_ms, max_spray_ms, image, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rec + (image_name, time.time()),
        )
    return jsonify(id=cur.lastrowid), 201


@app.get("/api/stops")
def list_stops():
    limit = min(max(request.args.get("limit", 50, type=int), 1), 500)
    with db() as con:
        rows = con.execute("SELECT * FROM stops ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return jsonify([row_to_dict(r) for r in rows])


@app.get("/api/summary")
def summary():
    with db() as con:
        t = con.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(spray_ms > 0),0) sprayed, COALESCE(SUM(spray_ms),0) spray_ms,"
            " COALESCE(SUM(max_spray_ms),0) base_ms, COALESCE(AVG(damage_pct),0) avg_damage,"
            " COUNT(DISTINCT robot_id) robots, MAX(ts) last_ts FROM stops"
        ).fetchone()
        states = {r["state"]: r["c"] for r in
                  con.execute("SELECT state, COUNT(*) c FROM stops GROUP BY state")}
        crops = {}
        for r in con.execute("SELECT label, damage_pct, spray_ms FROM stops"):
            c = crops.setdefault(crop_of(r["label"]), {"stops": 0, "damage_sum": 0.0, "spray_ms": 0})
            c["stops"] += 1
            c["damage_sum"] += r["damage_pct"]
            c["spray_ms"] += r["spray_ms"]
        trend = [r["damage_pct"] for r in con.execute(
            "SELECT damage_pct FROM stops ORDER BY id DESC LIMIT 100")][::-1]
        recent_severe = con.execute(
            "SELECT COUNT(*) FROM (SELECT state FROM stops ORDER BY id DESC LIMIT 10) WHERE state='SEVERE'"
        ).fetchone()[0]
    saving = round(100 * (1 - t["spray_ms"] / t["base_ms"]), 1) if t["base_ms"] else None
    return jsonify(
        stops=t["n"], sprayed=t["sprayed"], skipped=t["n"] - t["sprayed"],
        pump_seconds=round(t["spray_ms"] / 1000, 1), saving_pct=saving,
        avg_damage=round(t["avg_damage"], 1), robots=t["robots"], last_ts=t["last_ts"],
        states=states, trend=trend, recent_severe=recent_severe,
        crops=[{"crop": k, "stops": v["stops"], "avg_damage": round(v["damage_sum"] / v["stops"], 1),
                "pump_seconds": round(v["spray_ms"] / 1000, 1)} for k, v in sorted(crops.items())],
    )


@app.get("/api/export.csv")
def export_csv():
    with db() as con:
        rows = con.execute("SELECT * FROM stops ORDER BY id").fetchall()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id", "robot_id", "ts", "source", "label", "conf", "damage_pct",
                "state", "dose_ml_l", "spray_ms", "max_spray_ms"])
    for r in rows:
        w.writerow([r["id"], r["robot_id"], r["ts"], r["source"], r["label"], r["conf"],
                    r["damage_pct"], r["state"], r["dose_ml_l"], r["spray_ms"], r["max_spray_ms"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=agrospray_stops.csv"})


# ------------------------------------------------------------ static files
@app.get("/uploads/<path:name>")
def uploads(name):
    return send_from_directory(UPLOADS, name)


@app.get("/")
def index():
    return send_from_directory(DASHBOARD, "index.html")


init_db()

if __name__ == "__main__":
    if API_KEY == "agrospray-dev-key":
        print("WARNING: using the default API key. Set AGROSPRAY_KEY before exposing this server.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
