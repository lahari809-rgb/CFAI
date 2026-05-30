"""
Hospital Navigation Assistant — Python Flask Backend
Handles: Graph search algorithms, CSP validation, Bayesian risk assessment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import heapq
import time

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  GRAPH DATA
# ─────────────────────────────────────────────
NODES = {
    "entrance":   {"x": 60,  "y": 180, "label": "Entrance",   "type": "main",     "risk": 0.10, "wheelchair": True},
    "reception":  {"x": 150, "y": 180, "label": "Reception",  "type": "main",     "risk": 0.10, "wheelchair": True},
    "elevator":   {"x": 150, "y": 90,  "label": "Elevator",   "type": "util",     "risk": 0.15, "wheelchair": True},
    "stairs":     {"x": 150, "y": 270, "label": "Stairs",     "type": "util",     "risk": 0.20, "wheelchair": False},
    "radiology":  {"x": 270, "y": 60,  "label": "Radiology",  "type": "dept",     "risk": 0.25, "wheelchair": True},
    "pharmacy":   {"x": 270, "y": 180, "label": "Pharmacy",   "type": "dept",     "risk": 0.10, "wheelchair": True},
    "emergency":  {"x": 60,  "y": 300, "label": "Emergency",  "type": "critical", "risk": 0.60, "wheelchair": True},
    "icu":        {"x": 380, "y": 60,  "label": "ICU",        "type": "critical", "risk": 0.70, "wheelchair": True},
    "surgery":    {"x": 490, "y": 60,  "label": "Surgery",    "type": "critical", "risk": 0.50, "wheelchair": True},
    "ward_a":     {"x": 380, "y": 180, "label": "Ward A",     "type": "ward",     "risk": 0.20, "wheelchair": True},
    "ward_b":     {"x": 490, "y": 180, "label": "Ward B",     "type": "ward",     "risk": 0.20, "wheelchair": True},
    "cafeteria":  {"x": 380, "y": 300, "label": "Cafeteria",  "type": "main",     "risk": 0.05, "wheelchair": True},
    "lab":        {"x": 270, "y": 300, "label": "Lab",        "type": "dept",     "risk": 0.30, "wheelchair": True},
    "outpatient": {"x": 590, "y": 180, "label": "Outpatient", "type": "dept",     "risk": 0.15, "wheelchair": True},
    "physio":     {"x": 590, "y": 300, "label": "Physio",     "type": "dept",     "risk": 0.10, "wheelchair": True},
}

EDGES = [
    ("entrance",  "reception",  2),
    ("entrance",  "emergency",  3),
    ("reception", "elevator",   2),
    ("reception", "stairs",     2),
    ("reception", "pharmacy",   3),
    ("elevator",  "radiology",  2),
    ("elevator",  "ward_a",     3),
    ("elevator",  "icu",        4),
    ("stairs",    "pharmacy",   2),
    ("stairs",    "lab",        3),
    ("radiology", "icu",        2),
    ("icu",       "surgery",    2),
    ("pharmacy",  "ward_a",     2),
    ("pharmacy",  "lab",        2),
    ("ward_a",    "ward_b",     2),
    ("ward_a",    "cafeteria",  3),
    ("ward_b",    "surgery",    3),
    ("ward_b",    "outpatient", 2),
    ("cafeteria", "physio",     2),
    ("cafeteria", "lab",        2),
    ("outpatient","physio",     2),
    ("lab",       "emergency",  3),
]

# Build adjacency list
ADJ: dict[str, list[tuple[str, int]]] = {n: [] for n in NODES}
for a, b, w in EDGES:
    ADJ[a].append((b, w))
    ADJ[b].append((a, w))


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def heuristic(a: str, b: str) -> float:
    """Admissible Euclidean distance heuristic, scaled to match edge weights."""
    na, nb = NODES[a], NODES[b]
    return math.sqrt((na["x"] - nb["x"]) ** 2 + (na["y"] - nb["y"]) ** 2) / 55


def get_neighbours(node: str, wheelchair: bool, avoid_icu: bool) -> list[tuple[str, int]]:
    """Return filtered neighbours respecting active CSP constraints."""
    result = []
    for nb, w in ADJ[node]:
        if wheelchair and not NODES[nb]["wheelchair"]:
            continue
        if avoid_icu and nb == "icu":
            continue
        result.append((nb, w))
    return result


# ─────────────────────────────────────────────
#  SEARCH ALGORITHMS
# ─────────────────────────────────────────────

def algo_astar(start: str, end: str, wheelchair: bool, avoid_icu: bool, emergency: bool) -> dict | None:
    """
    A* Search — f(n) = g(n) + h(n).
    Admissible + consistent heuristic guarantees optimality.
    Uses a min-heap (priority queue) for efficiency.
    """
    # heap: (f, g, node, path)
    heap = [(heuristic(start, end), 0.0, start, [start])]
    visited: set[str] = set()
    expanded = 0

    while heap:
        f, g, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1

        if cur == end:
            return {"path": path, "cost": round(g, 1), "expanded": expanded}

        for nb, w in get_neighbours(cur, wheelchair, avoid_icu):
            if nb not in visited:
                risk_penalty = 0.0 if emergency else NODES[nb]["risk"]
                new_g = g + w + risk_penalty
                heapq.heappush(heap, (new_g + heuristic(nb, end), new_g, nb, path + [nb]))

    return None


def algo_bfs(start: str, end: str, wheelchair: bool, avoid_icu: bool) -> dict | None:
    """
    BFS — Level-by-level traversal.
    Guarantees fewest hops (unweighted shortest path). Complete on finite graphs.
    """
    from collections import deque
    queue: deque[tuple[str, list[str], float]] = deque([(start, [start], 0.0)])
    visited: set[str] = {start}
    expanded = 0

    while queue:
        cur, path, cost = queue.popleft()
        expanded += 1

        if cur == end:
            return {"path": path, "cost": round(cost, 1), "expanded": expanded}

        for nb, w in get_neighbours(cur, wheelchair, avoid_icu):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb], cost + w))

    return None


def algo_dfs(start: str, end: str, wheelchair: bool, avoid_icu: bool) -> dict | None:
    """
    DFS — Depth-first using an explicit stack.
    Memory-efficient O(depth). Non-optimal — may not find shortest path.
    """
    stack: list[tuple[str, list[str], float]] = [(start, [start], 0.0)]
    visited: set[str] = set()
    expanded = 0

    while stack:
        cur, path, cost = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1

        if cur == end:
            return {"path": path, "cost": round(cost, 1), "expanded": expanded}

        for nb, w in reversed(get_neighbours(cur, wheelchair, avoid_icu)):
            if nb not in visited:
                stack.append((nb, path + [nb], cost + w))

    return None


def algo_greedy(start: str, end: str, wheelchair: bool, avoid_icu: bool) -> dict | None:
    """
    Greedy Best-First — expands node with lowest h(n).
    Fast but non-optimal; may miss shorter paths.
    """
    heap = [(heuristic(start, end), start, [start], 0.0)]
    visited: set[str] = set()
    expanded = 0

    while heap:
        _, cur, path, cost = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1

        if cur == end:
            return {"path": path, "cost": round(cost, 1), "expanded": expanded}

        for nb, w in get_neighbours(cur, wheelchair, avoid_icu):
            if nb not in visited:
                heapq.heappush(heap, (heuristic(nb, end), nb, path + [nb], cost + w))

    return None


def algo_ucs(start: str, end: str, wheelchair: bool, avoid_icu: bool) -> dict | None:
    """
    UCS / Dijkstra — expands by cumulative cost g(n).
    Optimal on positive-cost graphs. No heuristic required.
    """
    heap: list[tuple[float, str, list[str]]] = [(0.0, start, [start])]
    visited: set[str] = set()
    expanded = 0

    while heap:
        cost, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1

        if cur == end:
            return {"path": path, "cost": round(cost, 1), "expanded": expanded}

        for nb, w in get_neighbours(cur, wheelchair, avoid_icu):
            if nb not in visited:
                heapq.heappush(heap, (cost + w, nb, path + [nb]))

    return None


# ─────────────────────────────────────────────
#  BAYESIAN RISK ENGINE
# ─────────────────────────────────────────────
def compute_bayesian_risk(path: list[str], patient_type: str) -> list[dict]:
    """
    Compute posterior risk for each node using Bayes' theorem:
        P(complication | evidence) = P(evidence | patient_type) * P(prior) / P(evidence)
    """
    priors = {"stable": 0.15, "critical": 0.70, "visiting": 0.05, "staff": 0.10}
    prior = priors.get(patient_type, 0.15)

    results = []
    for node_id in path:
        nd = NODES[node_id]
        likelihood = nd["risk"]
        # Bayes: posterior = (likelihood * prior) / [(likelihood * prior) + (1-likelihood)*(1-prior)]
        numerator   = likelihood * prior
        denominator = numerator + (1 - likelihood) * (1 - prior)
        posterior   = min(0.99, numerator / denominator if denominator > 0 else 0)
        results.append({
            "node":      node_id,
            "label":     nd["label"],
            "prior":     round(prior, 3),
            "likelihood": round(likelihood, 3),
            "posterior": round(posterior, 3),
            "risk_level": "high" if posterior > 0.5 else "medium" if posterior > 0.25 else "low",
        })
    return results


# ─────────────────────────────────────────────
#  CSP VALIDATOR
# ─────────────────────────────────────────────
def validate_csp(path: list[str], wheelchair: bool, avoid_icu: bool) -> list[dict]:
    """
    CSP Constraint Satisfaction check for each node on the path.
    Variables: nodes; Domains: {reachable rooms}; Constraints: accessibility + corridor rules.
    """
    report = []
    for node_id in path:
        nd = NODES[node_id]
        wc_ok  = not wheelchair or nd["wheelchair"]
        icu_ok = not avoid_icu  or node_id != "icu"
        report.append({
            "node":        node_id,
            "label":       nd["label"],
            "wheelchair":  nd["wheelchair"],
            "risk":        nd["risk"],
            "wc_ok":       wc_ok,
            "icu_ok":      icu_ok,
            "satisfies":   wc_ok and icu_ok,
        })
    return report


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/api/navigate", methods=["POST"])
def navigate():
    """
    Main navigation endpoint.
    POST body: { start, end, algorithm, wheelchair, avoid_icu, emergency, patient_type }
    Returns: path, cost, expanded, csp_report, bayesian_report, elapsed_ms
    """
    data         = request.get_json(force=True)
    start        = data.get("start", "entrance")
    end          = data.get("end",   "ward_a")
    algorithm    = data.get("algorithm", "astar")
    wheelchair   = bool(data.get("wheelchair", False))
    avoid_icu    = bool(data.get("avoid_icu",  False))
    emergency    = bool(data.get("emergency",  False))
    patient_type = data.get("patient_type", "stable")

    if start not in NODES or end not in NODES:
        return jsonify({"error": "Invalid start or end node."}), 400
    if start == end:
        return jsonify({"error": "Start and destination must differ."}), 400

    t0 = time.perf_counter()

    if   algorithm == "astar":  result = algo_astar(start, end, wheelchair, avoid_icu, emergency)
    elif algorithm == "bfs":    result = algo_bfs(start, end, wheelchair, avoid_icu)
    elif algorithm == "dfs":    result = algo_dfs(start, end, wheelchair, avoid_icu)
    elif algorithm == "greedy": result = algo_greedy(start, end, wheelchair, avoid_icu)
    elif algorithm == "ucs":    result = algo_ucs(start, end, wheelchair, avoid_icu)
    else:
        return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    if result is None:
        return jsonify({
            "found": False,
            "message": "No path found with current constraints.",
            "elapsed_ms": elapsed_ms,
        })

    path = result["path"]
    csp_report     = validate_csp(path, wheelchair, avoid_icu)
    bayesian_report = compute_bayesian_risk(path, patient_type)

    return jsonify({
        "found":       True,
        "path":        path,
        "path_labels": [NODES[n]["label"] for n in path],
        "cost":        result["cost"],
        "expanded":    result["expanded"],
        "hops":        len(path) - 1,
        "elapsed_ms":  elapsed_ms,
        "algorithm":   algorithm,
        "csp":         csp_report,
        "bayesian":    bayesian_report,
    })


@app.route("/api/graph", methods=["GET"])
def get_graph():
    """Return full graph data (nodes + edges) for frontend rendering."""
    return jsonify({"nodes": NODES, "edges": EDGES})


@app.route("/api/algorithms", methods=["GET"])
def get_algorithms():
    """Return metadata for all available search algorithms."""
    return jsonify([
        {"id": "astar",  "name": "A*",               "badge": "Optimal",  "desc": "f(n)=g(n)+h(n). Admissible heuristic guarantees optimality."},
        {"id": "bfs",    "name": "BFS",               "badge": "Complete", "desc": "Level-by-level. Guarantees fewest hops on unweighted graphs."},
        {"id": "dfs",    "name": "DFS",               "badge": "Memory↓",  "desc": "Depth-first stack. Memory-efficient, non-optimal."},
        {"id": "greedy", "name": "Greedy Best-First",  "badge": "Fast",     "desc": "Expands lowest h(n). Fast but may miss optimal paths."},
        {"id": "ucs",    "name": "UCS (Dijkstra)",     "badge": "Uniform",  "desc": "Expands by g(n). Optimal on positive-cost graphs."},
    ])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "nodes": len(NODES), "edges": len(EDGES)})


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Hospital Navigation Backend ===")
    print(f"  Graph: {len(NODES)} nodes, {len(EDGES)} edges")
    print("  Algorithms: A*, BFS, DFS, Greedy, UCS")
    print("  Endpoints: /api/navigate  /api/graph  /api/algorithms  /api/health")
    print("  Running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)