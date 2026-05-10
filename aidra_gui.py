# =============================================================================
#  AIDRA_GUI.PY  —  Simple Tkinter GUI
#
#  HOW TO RUN:
#    python aidra_gui.py
#
#  REQUIREMENTS:
#    - Python 3.x  (tkinter comes built-in)
#    - aidra_core.py in the same folder
#    - pip install scikit-learn numpy  (for the ML part)
#
#  NOTE: tkinter is built into Python on Windows and macOS.
#        On Linux run:  sudo apt install python3-tk
# =============================================================================

import tkinter as tk
from tkinter import scrolledtext, messagebox
import sys
import io
import threading   # so the GUI doesn't freeze while AI runs

# ── Import your AI core ───────────────────────────────────────────────────────
from aidra_core import (
    create_world, train_ml_models, predict_survival,
    run_csp, run_uncertainty, astar, bfs,
    rescue_victim, log_event,
    greedy, dfs, hill_climbing,
    BLOCKED, FIRE_ZONE, AFTERSHOCK,
    bayesian_blockage_prob
)

# ── Colours used in the grid ──────────────────────────────────────────────────
COLOURS = {
    "empty"    : "#1a2332",
    "blocked"  : "#5a1010",
    "fire"     : "#5a3200",
    "aftershock": "#3a3300",
    "path"     : "#0d4d3b",
    "base"     : "#1a3a6c",
    "hospital" : "#1a4a2a",
    "critical" : "#5a1020",
    "moderate" : "#5a3500",
    "minor"    : "#1a2d5a",
    "rescued"  : "#0d2a1a",
    "text"     : "#e0e0e0",
    "bg"       : "#1a1a2e",
    "panel"    : "#16213e",
    "red"      : "#e74c3c",
    "green"    : "#2ecc71",
    "yellow"   : "#f39c12",
    "blue"     : "#3498db",
    "accent"   : "#e94560",
}

CELL_SIZE = 36   # pixels per grid cell


# =============================================================================
#  HELPER: capture print() output from AI functions
# =============================================================================
def capture(func, *args, **kwargs):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


# =============================================================================
#  MAIN APPLICATION WINDOW
# =============================================================================
class AIDRAApp:
    def __init__(self, root):
        self.root  = root
        self.world = None
        self.ml    = None
        self.assignment = None

        root.title("AIDRA — Disaster Response Agent")
        root.configure(bg=COLOURS["bg"])
        root.resizable(True, True)

        self._build_ui()

    # ── Build the full UI layout ──────────────────────────────────────────────
    def _build_ui(self):
        # ── TOP HEADER ────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=COLOURS["panel"], pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🚑  AIDRA — Adaptive Intelligent Disaster Response Agent",
                 font=("Courier", 13, "bold"), bg=COLOURS["panel"],
                 fg=COLOURS["accent"]).pack(side="left", padx=12)
        self.status_lbl = tk.Label(hdr, text="Ready",
                                   font=("Courier", 10), bg=COLOURS["panel"],
                                   fg=COLOURS["text"])
        self.status_lbl.pack(side="right", padx=12)

        # ── MAIN AREA: left buttons | centre map | right output ───────────────
        main = tk.Frame(self.root, bg=COLOURS["bg"])
        main.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_buttons(main)   # left column
        self._build_map(main)       # centre column
        self._build_output(main)    # right column

    # ── LEFT: Phase buttons ───────────────────────────────────────────────────
    def _build_buttons(self, parent):
        frame = tk.Frame(parent, bg=COLOURS["panel"], width=160, padx=6, pady=8)
        frame.pack(side="left", fill="y", padx=(0, 6))
        frame.pack_propagate(False)

        tk.Label(frame, text="PHASES", font=("Courier", 9, "bold"),
                 bg=COLOURS["panel"], fg=COLOURS["accent"]).pack(pady=(0, 6))

        # Each button: (label, command, attribute name for later reference)
        phases = [
            ("1. Setup World",   self.phase_init,       "btn1"),
            ("2. Train ML",      self.phase_ml,         "btn2"),
            ("3. Fuzzy Logic",   self.phase_fuzzy,      "btn3"),
            ("4. CSP Allocate",  self.phase_csp,        "btn4"),
            ("5. Search Routes", self.phase_search,     "btn5"),
            ("6. Run Rescues",   self.phase_rescue,     "btn6"),
            ("7. Aftershock!",   self.phase_aftershock, "btn7"),
            ("8. Final Report",  self.phase_report,     "btn8"),
        ]

        self.buttons = {}
        for label, cmd, name in phases:
            btn = tk.Button(frame, text=label, command=cmd,
                            font=("Courier", 10), bg="#0f3460", fg="#cccccc",
                            activebackground="#1a5276", activeforeground="white",
                            relief="flat", anchor="w", padx=8, pady=5,
                            state="disabled" if name != "btn1" else "normal")
            btn.pack(fill="x", pady=2)
            self.buttons[name] = btn

        tk.Frame(frame, bg=COLOURS["panel"], height=10).pack()

        # Run All button
        tk.Button(frame, text="▶  Run All", command=self.run_all,
                  font=("Courier", 10, "bold"), bg="#6c1a1a", fg=COLOURS["accent"],
                  activebackground="#8b2222", relief="flat", pady=6
                  ).pack(fill="x", pady=2)

    # ── CENTRE: Grid map ──────────────────────────────────────────────────────
    def _build_map(self, parent):
        frame = tk.Frame(parent, bg=COLOURS["bg"])
        frame.pack(side="left", fill="y", padx=(0, 6))

        # Small legend above the map
        leg = tk.Frame(frame, bg=COLOURS["bg"])
        leg.pack(fill="x", pady=(0, 4))
        legend_items = [
            ("B=base", COLOURS["base"]),
            ("+=hosp", COLOURS["hospital"]),
            ("X=blocked", COLOURS["blocked"]),
            ("F=fire", COLOURS["fire"]),
            ("~=quake", COLOURS["aftershock"]),
            ("·=route", COLOURS["path"]),
        ]
        for txt, col in legend_items:
            tk.Label(leg, text="■ " + txt, font=("Courier", 8),
                     bg=COLOURS["bg"], fg=col).pack(side="left", padx=3)

        # Canvas to draw the 10×10 grid
        canvas_size = CELL_SIZE * 10 + 2
        self.canvas = tk.Canvas(frame, width=canvas_size, height=canvas_size,
                                bg="#0a0a1a", highlightthickness=1,
                                highlightbackground=COLOURS["panel"])
        self.canvas.pack()

        # Draw empty grid placeholder
        self._draw_empty_grid()

        # Victim list below the map
        tk.Label(frame, text="VICTIMS", font=("Courier", 9, "bold"),
                 bg=COLOURS["bg"], fg=COLOURS["accent"]).pack(pady=(8, 2))
        self.victim_frame = tk.Frame(frame, bg=COLOURS["bg"])
        self.victim_frame.pack(fill="x")
        tk.Label(self.victim_frame, text="Run Phase 1 to load",
                 font=("Courier", 9), bg=COLOURS["bg"],
                 fg="#555555").pack()

    def _draw_empty_grid(self):
        """Draw a blank 10×10 grid as placeholder."""
        self.canvas.delete("all")
        for r in range(10):
            for c in range(10):
                x1, y1 = c * CELL_SIZE + 1, r * CELL_SIZE + 1
                x2, y2 = x1 + CELL_SIZE - 2, y1 + CELL_SIZE - 2
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=COLOURS["empty"], outline="#0f1a2a")

    # ── RIGHT: Output text box ────────────────────────────────────────────────
    def _build_output(self, parent):
        frame = tk.Frame(parent, bg=COLOURS["bg"])
        frame.pack(side="left", fill="both", expand=True)

        tk.Label(frame, text="OUTPUT", font=("Courier", 9, "bold"),
                 bg=COLOURS["bg"], fg=COLOURS["accent"]).pack(anchor="w")

        self.output = scrolledtext.ScrolledText(
            frame,
            font=("Courier", 11),
            bg="#0a0a1a", fg="#c8d6e5",
            insertbackground="white",
            relief="flat", padx=10, pady=10,
            wrap="word", state="disabled"
        )
        self.output.pack(fill="both", expand=True)

    # =========================================================================
    #  GRID RENDERING
    # =========================================================================
    def render_grid(self, world, path=None):
        """Redraw the full 10×10 grid based on current world state."""
        path_set = set(path) if path else set()
        self.canvas.delete("all")

        for r in range(10):
            for c in range(10):
                cell_type = world["grid"][r][c]
                x1 = c * CELL_SIZE + 1
                y1 = r * CELL_SIZE + 1
                x2 = x1 + CELL_SIZE - 2
                y2 = y1 + CELL_SIZE - 2
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Work out colour and label for this cell
                colour, label, text_col = self._cell_style(
                    world, r, c, cell_type, path_set)

                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=colour, outline="#0f1a2a")
                if label:
                    self.canvas.create_text(cx, cy, text=label,
                                            font=("Courier", 8, "bold"),
                                            fill=text_col)

    def _cell_style(self, world, r, c, cell_type, path_set):
        """Return (background_colour, label_text, text_colour) for a cell."""
        # Check special positions first
        if (r, c) == world["base"]:
            return COLOURS["base"], "B", COLOURS["blue"]
        if any((r, c) == tuple(h["pos"]) for h in world["hospitals"]):
            return COLOURS["hospital"], "+", COLOURS["green"]

        # Check if a victim is here
        victim = next((v for v in world["victims"]
                       if v["row"] == r and v["col"] == c), None)
        if victim:
            if victim["rescued"]:
                return COLOURS["rescued"], victim["id"], COLOURS["green"]
            if victim["severity"] == "Critical":
                return COLOURS["critical"], victim["id"], COLOURS["red"]
            if victim["severity"] == "Moderate":
                return COLOURS["moderate"], victim["id"], COLOURS["yellow"]
            return COLOURS["minor"], victim["id"], COLOURS["blue"]

        # Route path highlight
        if (r, c) in path_set:
            return COLOURS["path"], "·", COLOURS["green"]

        # Terrain types
        if cell_type == BLOCKED:
            return COLOURS["blocked"], "X", COLOURS["red"]
        if cell_type == FIRE_ZONE:
            return COLOURS["fire"], "F", COLOURS["yellow"]
        if cell_type == AFTERSHOCK:
            return COLOURS["aftershock"], "~", COLOURS["yellow"]

        return COLOURS["empty"], "", COLOURS["text"]

    # =========================================================================
    #  VICTIM LIST (below map)
    # =========================================================================
    def render_victims(self, world):
        """Refresh the victim list below the map."""
        for w in self.victim_frame.winfo_children():
            w.destroy()

        victims = sorted(world["victims"], key=lambda v: v["priority"], reverse=True)
        for v in victims:
            row = tk.Frame(self.victim_frame, bg=COLOURS["panel"], pady=2, padx=4)
            row.pack(fill="x", pady=1)

            # Colour-code by severity
            col = (COLOURS["red"] if v["severity"] == "Critical" else
                   COLOURS["yellow"] if v["severity"] == "Moderate" else
                   COLOURS["blue"])
            status = "✅" if v["rescued"] else "⏳"
            surv   = f"{v['survival']:.0%}"

            tk.Label(row,
                     text=f"{status} {v['id']}  {v['name']:<14}  "
                          f"{v['severity']:<10}  survival={surv}",
                     font=("Courier", 9), bg=COLOURS["panel"], fg=col,
                     anchor="w").pack(fill="x")

    # =========================================================================
    #  OUTPUT TEXT BOX
    # =========================================================================
    def write(self, text):
        """Write text to the output box."""
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.config(state="disabled")
        self.output.see("end")

    def set_status(self, text):
        self.status_lbl.config(text=text)

    def enable_next(self, btn_name):
        """Enable the next phase button."""
        self.buttons[btn_name].config(state="normal")

    def mark_done(self, btn_name):
        """Turn a button green to show it's complete."""
        self.buttons[btn_name].config(bg="#0d3d1a", fg=COLOURS["green"])

    # =========================================================================
    #  PHASE FUNCTIONS
    #  Each one runs in a background thread so the GUI stays responsive.
    # =========================================================================

    def _run_in_thread(self, func):
        """Run func in a background thread (keeps GUI from freezing)."""
        threading.Thread(target=func, daemon=True).start()

    # ── Phase 1: Setup world ──────────────────────────────────────────────────
    def phase_init(self):
        self._run_in_thread(self._do_init)

    def _do_init(self):
        self.world = create_world()
        self.world["total_kits"] = self.world["kits_left"]
        self.ml = None
        self.assignment = None

        self.render_grid(self.world)
        self.render_victims(self.world)
        self.mark_done("btn1")
        self.enable_next("btn2")
        self.set_status("Phase 1 done | Time: 0 min | Kits: 10 | Rescued: 0/5")
        self.write(
            "World created — Kashmir 2005, Muzaffarabad\n\n"
            "Victims:\n"
            "  V1  Aisha Bibi     Critical   (1,1)\n"
            "  V2  Tariq Ahmed    Critical   (1,6)  ← near fire zone\n"
            "  V3  Fatima Malik   Moderate   (6,3)\n"
            "  V4  Shahid Khan    Moderate   (5,7)\n"
            "  V5  Zara Noor      Minor      (8,5)\n\n"
            "Resources:\n"
            "  AMB-1    Ambulance    (max 2 victims)\n"
            "  AMB-2    Ambulance    (max 2 victims)\n"
            "  TEAM-1   Rescue Team  (max 1 victim)\n\n"
            "Medical kits: 10\n"
            "Hospitals: MED-1 (0,9)   MED-2 (9,9)\n\n"
            "Map loaded. Run Phase 2 next."
        )

    # ── Phase 2: Train ML models ──────────────────────────────────────────────
    def phase_ml(self):
        self._run_in_thread(self._do_ml)

    def _do_ml(self):
        self.write("Training ML models... (this takes a moment)")
        ml, _ = capture(train_ml_models, self.world)
        self.ml = ml

        lines = ["ML models trained on 800 synthetic rescue samples.\n"]
        lines.append(f"  {'Model':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7}")
        lines.append("  " + "-"*52)
        for name, m in ml["metrics"].items():
            lines.append(f"  {name:<16} {m['accuracy']:>9.1%} {m['precision']:>10.1%}"
                         f" {m['recall']:>7.1%} {m['f1']:>7.1%}")

        lines.append("\nSurvival predictions:\n")
        lines.append(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'Survival':>8}  Action")
        lines.append("  " + "-"*54)
        for v in sorted(self.world["victims"], key=lambda x: x["priority"], reverse=True):
            prob = predict_survival(ml, v, self.world["time"], self.world)
            v["survival"] = prob
            action = "RESCUE NOW" if prob < 0.6 else ("Rescue soon" if prob < 0.8 else "Stable")
            lines.append(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10}"
                         f" {prob:>8.0%}  {action}")

        self.render_victims(self.world)
        self.mark_done("btn2")
        self.enable_next("btn3")
        self.set_status("Phase 2 done — ML models trained")
        self.write("\n".join(lines))

    # ── Phase 3: Fuzzy + Bayesian ─────────────────────────────────────────────
    def phase_fuzzy(self):
        self._run_in_thread(self._do_fuzzy)

    def _do_fuzzy(self):
        urgency, _ = capture(run_uncertainty, self.world, label="Initial (t=0)")

        lines = ["Fuzzy urgency scores (0 = low, 10 = max urgency):\n"]
        lines.append(f"  {'ID':<4} {'Name':<15} {'Severity':<10} {'Score':>6}  Bar")
        lines.append("  " + "-"*52)
        for vid, u in urgency.items():
            v = next(x for x in self.world["victims"] if x["id"] == vid)
            bar = "█" * int(u) + "░" * (10 - int(u))
            lines.append(f"  {vid:<4} {v['name']:<15} {v['severity']:<10}"
                         f" {u:>6.1f}  [{bar}]")

        lines.append("\nBayesian road blockage probability:\n")
        for zone, label in [("EMPTY", "Normal roads"),
                             ("AFTERSHOCK", "Aftershock zones"),
                             ("FIRE_ZONE",  "Fire zones")]:
            p = bayesian_blockage_prob(zone, self.world["aftershocks"], self.world["time"])
            bar = "█" * int(p * 10) + "░" * (10 - int(p * 10))
            lines.append(f"  {label:<18} [{bar}]  {p:.0%}")

        self.mark_done("btn3")
        self.enable_next("btn4")
        self.set_status("Phase 3 done — Fuzzy & Bayesian analysis complete")
        self.write("\n".join(lines))

    # ── Phase 4: CSP ─────────────────────────────────────────────────────────
    def phase_csp(self):
        self._run_in_thread(self._do_csp)

    def _do_csp(self):
        assignment, _ = capture(run_csp, self.world, compare=True)
        self.assignment = assignment

        lines = ["CSP solved — Backtracking + MRV heuristic\n"]
        lines.append("Constraints: max 2 victims/ambulance, max 1/rescue team\n")
        lines.append(f"  {'Victim':<6} {'Name':<15} {'Severity':<10} Assigned To")
        lines.append("  " + "-"*46)
        for v in sorted(self.world["victims"], key=lambda x: x["priority"], reverse=True):
            rid = assignment.get(v["id"], "WAIT")
            lines.append(f"  {v['id']:<6} {v['name']:<15} {v['severity']:<10} {rid}")

        lines.append("\nResource loads:")
        for res in self.world["resources"]:
            assigned = [vid for vid, rid in assignment.items() if rid == res["id"]]
            bar = "█" * len(assigned) + "░" * (res["max"] - len(assigned))
            lines.append(f"  {res['id']:<8} [{bar}]  {len(assigned)}/{res['max']}  {assigned}")

        self.render_victims(self.world)
        self.mark_done("btn4")
        self.enable_next("btn5")
        self.set_status("Phase 4 done — resources allocated")
        self.write("\n".join(lines))

    # ── Phase 5: Search algorithms ────────────────────────────────────────────
    def phase_search(self):
        self._run_in_thread(self._do_search)

    def _do_search(self):
        grid, base = self.world["grid"], self.world["base"]
        v1 = next(v for v in self.world["victims"] if v["id"] == "V1")
        v2 = next(v for v in self.world["victims"] if v["id"] == "V2")

        algos = [
            bfs(grid, base, (v1["row"], v1["col"])),
            dfs(grid, base, (v1["row"], v1["col"])),
            greedy(grid, base, (v1["row"], v1["col"])),
            astar(grid, base, (v1["row"], v1["col"])),
            hill_climbing(grid, base, (v1["row"], v1["col"])),
        ]

        lines = ["Algorithm comparison — Base → V1:\n"]
        lines.append(f"  {'Algorithm':<16} {'Steps':>6} {'Cost':>6} {'Expanded':>9} {'Risk':>6}")
        lines.append("  " + "-"*50)
        for r in algos:
            if r["path"]:
                lines.append(f"  {r['name']:<16} {len(r['path']):>6} {r['cost']:>6}"
                             f" {r['expanded']:>9} {r['risk']:>6}")
            else:
                lines.append(f"  {r['name']:<16}  NO PATH")

        ra = astar(grid, base, (v2["row"], v2["col"]))
        rg = greedy(grid, base, (v2["row"], v2["col"]))
        lines.append(f"\nKey trade-off — V2 (near fire zone):")
        lines.append(f"  A*     cost={ra['cost']}  risk={ra['risk']}  ← CHOSEN")
        lines.append(f"  Greedy cost={rg['cost']}  risk={rg['risk']}  ← more risk")
        lines.append(f"\n  Map shows A* route to V2 (green dots).")

        # Show the A* path on the map
        self.render_grid(self.world, path=ra["path"] or [])
        self.mark_done("btn5")
        self.enable_next("btn6")
        self.set_status("Phase 5 done — A* route shown on map")
        self.write("\n".join(lines))

    # ── Phase 6: Execute rescues ──────────────────────────────────────────────
    def phase_rescue(self):
        self._run_in_thread(self._do_rescue)

    def _do_rescue(self):
        if not self.assignment:
            self.write("Run Phase 4 (CSP) first!")
            return

        grid, base = self.world["grid"], self.world["base"]
        order = sorted(self.world["victims"], key=lambda v: v["priority"], reverse=True)
        lines = ["Rescues — Critical first, then Moderate, then Minor:\n"]

        for v in order:
            if v["rescued"]:
                continue
            rid = self.assignment.get(v["id"], "WAIT")
            if rid == "WAIT":
                free = [r for r in self.world["resources"] if r["is_free"]]
                rid  = free[0]["id"] if free else "WAIT"
            if rid == "WAIT":
                lines.append(f"  {v['id']} {v['name']} — no resource free")
                continue

            route = astar(grid, base, (v["row"], v["col"]))
            if not route["path"]:
                route = bfs(grid, base, (v["row"], v["col"]))  # fallback
            if not route["path"]:
                lines.append(f"  ❌ {v['id']} {v['name']} — unreachable")
                continue

            rescue_victim(self.world, v, rid, route)
            res = next(r for r in self.world["resources"] if r["id"] == rid)
            res["is_free"] = True
            res["load"] = 0
            lines.append(f"  ✅ {v['id']}  {v['name']:<15} "
                         f"via {rid:<8}  {v['rescue_time']}min  "
                         f"survival={v['survival']:.0%}")

        rescued = len([v for v in self.world["victims"] if v["rescued"]])
        self.render_grid(self.world)
        self.render_victims(self.world)
        self.mark_done("btn6")
        self.enable_next("btn7")
        self.set_status(f"Phase 6 done | Time: {self.world['time']} min | "
                        f"Rescued: {rescued}/5 | Kits: {self.world['kits_left']}")
        self.write("\n".join(lines))

    # ── Phase 7: Aftershock ───────────────────────────────────────────────────
    def phase_aftershock(self):
        self._run_in_thread(self._do_aftershock)

    def _do_aftershock(self):
        for r, c in [(4, 5), (3, 7)]:
            if self.world["grid"][r][c] != "BLOCKED":
                self.world["grid"][r][c] = "BLOCKED"
        self.world["aftershocks"] += 1
        log_event(self.world, "AFTERSHOCK! Roads (4,5),(3,7) blocked — replanning")

        lines = ["⚡ Aftershock! New blockages: (4,5), (3,7)\n"]
        waiting = [v for v in self.world["victims"] if not v["rescued"]]
        if not waiting:
            lines.append("All victims already rescued — no replanning needed.")
        for v in waiting:
            route = astar(self.world["grid"], self.world["base"],
                          (v["row"], v["col"]))
            if route["path"]:
                free = [r for r in self.world["resources"] if r["is_free"]]
                rid  = free[0]["id"] if free else "TEAM-1"
                rescue_victim(self.world, v, rid, route)
                lines.append(f"  ✅ {v['id']} {v['name']} — replanned and rescued")
            else:
                lines.append(f"  ❌ {v['id']} {v['name']} — unreachable after quake")

        self.render_grid(self.world)
        self.render_victims(self.world)
        self.mark_done("btn7")
        self.enable_next("btn8")
        self.set_status(f"Phase 7 done | Aftershocks: {self.world['aftershocks']}")
        self.write("\n".join(lines))

    # ── Phase 8: Final report ─────────────────────────────────────────────────
    def phase_report(self):
        self._run_in_thread(self._do_report)

    def _do_report(self):
        w       = self.world
        rescued = [v for v in w["victims"] if v["rescued"]]
        times   = [v["rescue_time"] for v in rescued if v["rescue_time"]]
        avg     = sum(times) / len(times) if times else 0
        kits    = w.get("total_kits", 10) - w["kits_left"]

        lines = ["=" * 50, "  FINAL KPI REPORT", "=" * 50, ""]
        lines.append(f"  Victims rescued    : {len(rescued)} / {len(w['victims'])}")
        lines.append(f"  Avg rescue time    : {avg:.1f} min")
        lines.append(f"  Total time elapsed : {w['time']} min")
        lines.append(f"  Medical kits used  : {kits} / {w.get('total_kits', 10)}")
        lines.append(f"  Aftershocks handled: {w['aftershocks']}")
        lines.append("")
        lines.append(f"  {'ID':<4} {'Name':<15} {'Severity':<10} "
                     f"{'Status':<14} {'Time':>6} {'Survival':>9}")
        lines.append("  " + "-"*60)
        for v in w["victims"]:
            status = "RESCUED ✅" if v["rescued"] else "NOT REACHED ❌"
            t = f"{v['rescue_time']}min" if v["rescue_time"] else "—"
            lines.append(f"  {v['id']:<4} {v['name']:<15} {v['severity']:<10} "
                         f"{status:<14} {t:>6} {v['survival']:>8.0%}")

        if self.ml:
            lines.append("\n  ML Results:")
            lines.append(f"  {'Model':<16} {'Accuracy':>9} {'F1':>8}")
            lines.append("  " + "-"*34)
            for name, m in self.ml["metrics"].items():
                lines.append(f"  {name:<16} {m['accuracy']:>9.1%} {m['f1']:>8.1%}")

        lines.append("\n  Event Log:")
        for e in w["log"]:
            lines.append("  " + e)

        self.mark_done("btn8")
        self.set_status("Simulation complete!")
        self.write("\n".join(lines))

    # ── Run All ───────────────────────────────────────────────────────────────
    def run_all(self):
        """Run all 8 phases in sequence."""
        def _all():
            for fn in [self._do_init, self._do_ml, self._do_fuzzy,
                       self._do_csp, self._do_search, self._do_rescue,
                       self._do_aftershock, self._do_report]:
                fn()
                self.root.update()  # refresh UI between phases
        threading.Thread(target=_all, daemon=True).start()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = AIDRAApp(root)
    root.mainloop()