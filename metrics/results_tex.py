#!/usr/bin/env python3
"""
LaTeX renderer of the results — from metrics/out/results.json to a .tex file.

Why it exists, in one line: `results.md` is read, not cited. A LaTeX report
needs LaTeX tables and, above all, **numbers callable from running text**: if
the report says "the measured order is 2.00" by hand, that 2.00 is already
dead: it survives the next `make_results.py` without warning.

It therefore generates THREE files in `metrics/out/tex/`:

  metrics_macros.tex      one macro per scalar (\\resOrderMidpoint, ...). This
                          is the piece that makes the update automatic: the
                          report writes $\\resOrderMidpoint$ and the number
                          follows the code on its own.
  metrics_body.tex        sections and tables, WITHOUT a preamble: the file to
                          \\input{} inside Report.tex once the structure of the
                          report is settled.
  metrics_standalone.tex  minimal preamble + the two files above: it compiles on
                          its own in the report repository, without touching
                          Report.tex.

The order of the sections follows the report, not the code: model and
discretisation, NLP structure, derivatives, optimality conditions, regularity of
the solution, reformulations, closed-loop campaigns. Every section carries a
`\\resNote{...}` saying which section of the report it is meant for: at
integration time that macro is emptied and the notes all disappear together.

The .tex content is in English because the report is: it has to be
copy-pasteable without translation.

Usage:
    python3 metrics/results_tex.py                     # from metrics/out/results.json
    python3 metrics/results_tex.py --results other.json --out /tmp/tex
    python3 metrics/results_tex.py --check             # syntax check only

It is also invoked at the end of `metrics/make_results.py`, so a single
`python3 metrics/make_results.py` updates measurements and LaTeX in one go.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# ---------------------------------------------------------------------------
# Number formatting
#
# Rule: the numeric macros expand to content to be used INSIDE $...$
# (hence "1.29\times 10^{-10}" and not "$1.29\times 10^{-10}$"), the macros
# testuali in testo normale. E' dichiarata in testa a metrics_macros.tex.
# ---------------------------------------------------------------------------
DASH = "---"


def sci(v, d: int = 2) -> str:
    """Scientific notation in LaTeX form, without math-mode delimiters."""
    if v is None:
        return DASH
    v = float(v)
    if v == 0.0:
        return "0"
    mant, exp = f"{v:.{d}e}".split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


def fx(v, d: int = 2) -> str:
    """Virgola fissa."""
    return DASH if v is None else f"{float(v):.{d}f}"


def smart(v, sig: int = 3) -> str:
    """Fixed point if the number reads well that way, scientific otherwise."""
    if v is None:
        return DASH
    v = float(v)
    a = abs(v)
    if a == 0.0:
        return "0"
    if 1e-3 <= a < 1e5:
        d = max(0, sig - 1 - int(math.floor(math.log10(a))))
        return f"{v:.{d}f}"
    return sci(v, sig - 1)


def pc(v, d: int = 1) -> str:
    """Frazione -> percentuale."""
    return DASH if v is None else f"{100.0 * float(v):.{d}f}"


def m(s: str) -> str:
    """
    Wraps an already formatted value in $...$.

    Needed because sci()/smart() produce math-mode content
    ("1.74\\times 10^{-2}"): inside a table cell it has to be delimited, or LaTeX
    stops at \\times outside $. The macros instead stay bare, because in the
    report text they are already written inside $...$.
    """
    return s if s == DASH else f"${s}$"


# The motion regimes arrive from the JSON with the Italian names of
# make_results.py; the document is in English because the report is.
REGIMES_EN = {
    "nominale (vx=0.2, w=0.3)": "nominal ($v_x=0.2$, $\\omega=0.3$)",
    "con deriva laterale": "with lateral drift",
    "rotazione rapida (w=1.0)": "fast rotation ($\\omega=1.0$)",
}


def regime(name: str) -> str:
    return REGIMES_EN.get(name, esc(name))


# The outcomes of the satellite scripts are Italian strings, sometimes with
# markdown emphasis (**...**), which would stay literal in LaTeX.
OUTCOMES_EN = {
    "vincolo inattivo": "constraint inactive",
    "tightening efficace": "tightening effective",
    "inammissibile": "infeasible",
    "efficace": "effective",
}


# The regimes of the solver comparison also arrive in Italian.
SOLVER_REGIMES_EN = {
    "penalty (ostacoli nel costo)": "penalty (obstacles in the cost)",
    "l1 (ostacoli come vincoli)": "$\\ell^1$ (obstacles as constraints)",
}


def solver_regime(name: str) -> str:
    return SOLVER_REGIMES_EN.get(str(name).strip(), esc(name))


def outcome(name: str) -> str:
    raw = str(name).replace("*", "").strip()
    en = OUTCOMES_EN.get(raw.lower())
    return esc(en) if en else esc(raw)


def yesno(b) -> str:
    return DASH if b is None else ("yes" if b else "no")


def esc(s) -> str:
    """Escape of text that ends up in LaTeX (paths, bag names, ...)."""
    if s is None:
        return DASH
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        out = out.replace(a, b)
    return out


def tt(s) -> str:
    """Monospaced text, escaped."""
    return r"\texttt{" + esc(s) + "}"


# ---------------------------------------------------------------------------
# Raccolta delle macro
# ---------------------------------------------------------------------------
class Macros:
    """
    Accumulates the name/value pairs that become \\resdef{...}{...}.

    The name must consist of letters only: in LaTeX a macro cannot contain digits
    or underscores. `add` checks it instead of producing a file that does not
    compila.
    """

    _OK = re.compile(r"^res[A-Za-z]+$")

    def __init__(self) -> None:
        # The deployed parameters are needed by the sections fed by the
        # satellite scripts too, which do not receive results.json.
        self.params: dict = {}
        self._groups: list[tuple[str, list[tuple[str, str, str]]]] = []
        self._seen: set[str] = set()
        self._cur: list[tuple[str, str, str]] | None = None

    def group(self, title: str) -> None:
        self._cur = []
        self._groups.append((title, self._cur))

    def add(self, name: str, value: str, comment: str = "") -> str:
        if self._cur is None:
            self.group("misc")
        if not self._OK.match(name):
            raise ValueError(f"invalid macro name for LaTeX: {name!r} "
                             "(prefix 'res' + letters only)")
        if name in self._seen:
            raise ValueError(f"duplicate macro: {name!r}")
        self._seen.add(name)
        self._cur.append((name, value, comment))
        return "\\" + name

    def render(self, meta: dict) -> str:
        L = [
            "% " + "=" * 72,
            "% metrics_macros.tex — AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND",
            "%",
            "%   regenerate with: python3 metrics/make_results.py",
            "%             or with: python3 metrics/results_tex.py",
            "%",
            f"%   commit {meta.get('git_commit','')[:10]} on {meta.get('git_branch','')}"
            f"   ({meta.get('data_utc','')})",
            "%",
            "% CONVENTION",
            "%   - the NUMERIC macros expand to math-mode content:",
            "%       one writes  $\\resOrderMidpoint$  and not  \\resOrderMidpoint",
            "%   - the TEXT macros (branch, commit, profile, bag) go in text mode.",
            "%",
            "% USE IN THE REPORT",
            "%   \\input{metrics_macros}   in the preamble, then in the body:",
            "%       ``the measured order is $\\resOrderMidpoint$, against",
            "%         $\\resOrderEuler$ for forward Euler''",
            "%   The number follows the code: no digit copied by hand.",
            "% " + "=" * 72,
            "",
            r"\providecommand{\resdef}[2]{\expandafter\def\csname #1\endcsname{#2}}",
            "",
            "% Table lookup. Each one lives in its own file under tab/, so the",
            "% report includes it where it wants and it stays live at every regeneration.",
            "% Compiling from another folder only redefines \\restabdir:",
            "%     \\renewcommand{\\restabdir}{Metrics/tab}",
            r"\providecommand{\restabdir}{tab}",
            r"\providecommand{\restab}[1]{\input{\restabdir/#1}}",
            "",
        ]
        for title, items in self._groups:
            if not items:
                continue
            L.append(f"% --- {title} " + "-" * max(0, 68 - len(title)))
            width = max(len(n) for n, _, _ in items)
            for name, value, comment in items:
                line = f"\\resdef{{{name}}}{{{value}}}"
                if comment:
                    pad = " " * max(1, width + 14 - len(line))
                    line += f"{pad}% {comment}"
                L.append(line)
            L.append("")
        return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------
# The tables are not inlined in the body: each one ends up in a file of its own,
# under tab/, and both the staging document and Report_metrics.tex pull it in
# with \restab{name}. One source, two consumers, and the
# report stays live when it is regenerated.
TABLES: dict[str, list[str]] = {}


def _tabname(label: str) -> str:
    """res:tab:solvercmp -> solvercmp ; res:tab:horizon:narrowgap -> horizon_narrowgap"""
    name = label.replace("res:tab:", "").replace("res:", "")
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def table(spec: str, header: list[str], rows: list[list[str]], caption: str,
          label: str, small: bool = True, note: str = "") -> list[str]:
    """Register the table body and return the float that includes it.

    THE GENERATED FILE CONTAINS ONLY THE TABULAR — no float, no caption, no
    label. The caption lives in the document that includes the table, so it can be
    rewritten by hand where the writing happens, instead of in this generator. A
    regenerated file therefore cannot wipe out a caption written in the report,
    which was how captions used to get lost.

    `metrics_body.tex` remains self-contained because the float this function
    returns carries the default caption with it.
    """
    T = [f"\\begin{{tabular}}{{{spec}}}"]
    T.append(r"  \toprule")
    T.append("  " + " & ".join(header) + r" \\")
    T.append(r"  \midrule")
    for r in rows:
        T.append("  " + " & ".join(r) + r" \\")
    T.append(r"  \bottomrule")
    T.append(r"\end{tabular}")
    if note:
        T.append(r"\\[2pt] {\footnotesize " + note + "}")
    nome = _tabname(label)
    TABLES[nome] = T

    F = [r"\begin{table}[htbp]", r"  \centering"]
    if small:
        F.append(r"  \small")
    F.append(f"  \\caption{{{caption}}}")
    F.append(f"  \\label{{{label}}}")
    F.append("  \\restab{" + nome + "}")
    F.append(r"\end{table}")
    F.append("")
    return F


# ---------------------------------------------------------------------------
# Sections of the body, in the order they will be needed in the report
# ---------------------------------------------------------------------------
def sec_provenance(res: dict, M: Macros) -> list[str]:
    m = res["meta"]
    p = m["parametri_chiave"]
    M.params = dict(p)
    M.group("provenance and profile")
    commit = M.add("resCommit", esc(m.get("git_commit", "")[:10]), "commit corto")
    branch = M.add("resBranch", esc(m.get("git_branch", "")), "branch")
    profile = M.add("resProfile", tt(os.path.basename(m.get("profilo", ""))), "file YAML")
    date = M.add("resDate", esc(m.get("data_utc", "")[:10]), "date of the run")
    M.add("resCasadi", esc(m.get("casadi", "")))
    M.add("resNumpy", esc(m.get("numpy", "")))
    M.add("resPython", esc(m.get("python", "")))
    bag = M.add("resBag", tt(os.path.basename(str(m.get("bag", "")).rstrip("/"))), "bag used")

    M.group("deployed parameters (from the profile, not copied by hand)")
    N = M.add("resN", str(p["N"]), "orizzonte in passi")
    dt = M.add("resDt", fx(p["dt"], 2), "sampling step [s]")
    M.add("resHorizonSeconds", fx(p["N"] * p["dt"], 2), "N*dt [s]")
    M.add("resVref", fx(p["v_ref"], 2), "cruise speed [m/s]")
    M.add("resVxMax", fx(p["vx_max"], 2))
    if "vx_min" in p:
        # negative: the report uses it inside an interval, so the sign is needed
        M.add("resVxMin", fx(p["vx_min"], 2), "reverse limit [m/s]")
        M.add("resVxMinAbs", fx(abs(p["vx_min"]), 2), "magnitude of the reverse limit")
    M.add("resVyMax", fx(p["vy_max"], 2))
    M.add("resOmegaMax", fx(p["omega_max"], 2))
    M.add("resWobsDep", smart(p["W_obs_sigmoid"]), "weight of the barrier")
    M.add("resObsRDep", fx(p["obs_r"], 2), "safety radius [m]")
    if "obs_alpha" in p:
        M.add("resAobsDep", smart(p["obs_alpha"]), "slope of the barrier")
        M.add("resObsCheckR", fx(p["obs_check_radius"], 1),
              "obstacle search radius [m]")
        M.add("resMobs", str(int(p["max_obs_constraints"])),
              "fixed number of obstacle terms")
    M.add("resTauV", smart(p["tau_v"]), "time constant [s]")
    M.add("resIntegrator", esc(p["integrator"]))
    M.add("resPathMode", esc(p["path_mode"]))
    M.add("resTerminalMode", esc(p["terminal_constraint"]))

    dirty = m.get("git_albero_sporco", False)
    L = [
        r"\resSec{Measured quantities of the optimization problem}",
        r"\label{res:top}",
        "",
        r"\resNote{This file is generated by \texttt{metrics/results\_tex.py} from "
        r"\texttt{metrics/out/results.json}; every number below is computed by the same "
        r"modules the deployed planner imports. Do not edit it by hand: re-run "
        r"\texttt{python3 metrics/make\_results.py}. Each block carries a note naming the "
        r"section of the report it is meant to feed; emptying \texttt{\textbackslash resNote} "
        r"removes all of them at once.}",
        "",
        r"\resSubsec{Provenance}",
        "",
        f"All quantities in this document were produced on {date} from commit "
        f"\\texttt{{{commit}}} of branch \\texttt{{{branch}}}, with the deployed profile "
        f"{profile} and the recorded run {bag}. "
        f"The measurement stack is CasADi~\\resCasadi, NumPy~\\resNumpy, "
        f"Python~\\resPython.",
        "",
        f"The configuration under test is $N={N}$, $\\Delta t={dt}$~s "
        f"(a $\\resHorizonSeconds$~s horizon), cruise speed "
        f"$v_{{\\mathrm{{ref}}}}=\\resVref$~m/s, input envelope "
        f"$(\\resVxMax,\\ \\resVyMax,\\ \\resOmegaMax)$ in m/s and rad/s, "
        f"barrier $(\\Wobs,\\robs)=(\\resWobsDep,\\ \\resObsRDep$~m$)$, "
        f"integrator \\texttt{{\\resIntegrator}}, reference mode "
        f"\\texttt{{\\resPathMode}}, terminal constraint \\texttt{{\\resTerminalMode}}.",
        "",
    ]
    if dirty:
        L += [
            r"\begin{center}\fbox{\begin{minipage}{0.92\linewidth}\small",
            r"\textbf{Warning --- dirty working tree.} These numbers were produced with "
            r"uncommitted changes in the repository, so they are \emph{not} reproducible "
            r"from the commit named above. Commit the tree and re-run before quoting them "
            r"in the report.",
            r"\end{minipage}}\end{center}",
            "",
        ]
    return L


def sec_discretisation(res: dict, M: Macros) -> list[str]:
    d = res.get("classe1", {}).get("integratore")
    b = res.get("classe2", {}).get("integratore_bag")
    if not d:
        return []
    M.group("discretisation (truncation order)")
    reg = d["regimi"]
    nom_key = next(iter(reg))
    nom = reg[nom_key]
    oe = M.add("resOrderEuler", fx(nom["ordine_euler"], 2), "ordine misurato, Euler")
    om = M.add("resOrderMidpoint", fx(nom["ordine_midpoint"], 2), "ordine misurato, punto medio")
    dep = d["al_dt_deployato"]
    ee = M.add("resErrEulerDep", sci(dep["errore_euler_m"]), "errore a dt deployato [m], arco sintetico")
    em = M.add("resErrMidpointDep", sci(dep["errore_midpoint_m"]))
    gain = M.add("resIntegratorGain", f"{dep['guadagno']:.0f}", "rapporto Euler/punto medio")
    Tsyn = M.add("resOrderWindow", fx(d.get("orizzonte_s", 3.0), 2), "window of the synthetic test [s]")

    L = [
        r"\resSubsec{Model discretisation: truncation order}",
        r"\label{res:disc}",
        "",
        r"\resNote{Feeds Report \texttt{sec:discretization}. Two measurements: the order of "
        r"the scheme on a closed-form arc, and the error it actually makes on horizons the "
        r"MPC really solved.}",
        "",
        f"The global integration error was fitted against the step size on three motion "
        f"regimes, over a ${Tsyn}$~s window, with the exact solution available in closed "
        f"form. The measured orders are ${oe}$ for forward Euler and ${om}$ for the "
        f"mid-point rule (Table~\\ref{{res:tab:order}}), i.e.\\ exactly the first and second "
        f"order predicted by the truncation analysis. At the deployed "
        f"$\\Delta t=\\resDt$~s the two differ by ${ee}$~m against ${em}$~m, a factor "
        f"${gain}$. The step grid is built by halving from the deployed step, so the first "
        f"row of the fit is the step the controller actually uses.",
        "",
    ]
    rows = [[regime(k), fx(v["ordine_euler"], 2), fx(v["ordine_midpoint"], 2)]
            for k, v in reg.items()]
    L += table("lrr", ["motion regime", "order, Euler", "order, mid-point"], rows,
               "Fitted global truncation order of the two integrators, by motion regime "
               "(log--log fit of the global error against the step size, on a "
               "constant-velocity arc whose exact solution is known).",
               "res:tab:order")

    bl = res.get("classe1", {}).get("bersaglio_locale")
    if bl:
        M.group("local target: the three metrics (figure §3.1)")
        M.add("resFigWorld", esc(bl["mondo"]), "world of the figure")
        M.add("resFigProjGeo", fx(bl["geo_proiezione"], 1),
              "geodesic of the target chosen by projection [m]")
        M.add("resFigEuclGeo", fx(bl["geo_argmin_euclideo"], 1),
              "geodesic of the target chosen by the euclidean argmin [m]")
        M.add("resFigGeoGeo", fx(bl["geo_geodetica"], 1),
              "geodesic of the target chosen by the geodesic [m]")

    if not b:
        return L

    # --- the same measurement on REAL horizons ---------------------------
    M.group("integrator on real horizons (bag)")
    bd = b["al_dt_deployato"]
    bc = M.add("resIntegBagCycles", str(b["cicli_usati"]), "orizzonti ri-risolti")
    bbag = M.add("resIntegBagName", esc(b["bag"]), "bag usata")
    be = M.add("resIntegBagEuler", sci(bd["errore_euler_m"]), "errore Euler, orizzonti veri [m]")
    bm = M.add("resIntegBagMidpoint", sci(bd["errore_midpoint_m"]))
    bep = M.add("resIntegBagEulerTail", sci(bd["p95_euler_m"]))
    bmp = M.add("resIntegBagMidpointTail", sci(bd["p95_midpoint_m"]))
    bg = M.add("resIntegBagGain", f"{bd['guadagno']:.0f}", "rapporto sugli orizzonti veri")
    boe = M.add("resIntegBagOrderEuler", fx(b["ordine"]["euler"], 2))
    bom = M.add("resIntegBagOrderMidpoint", fx(b["ordine"]["midpoint"], 2))
    bw = M.add("resIntegBagOmega", fx(b["omega_ptp_mediano"], 2), "excursion of omega [rad/s]")
    bl = M.add("resLagFloor", sci(b["lag_mediano_m"]), "displacement neglected by the lag [m]")

    L += [
        "",
        f"The same comparison was then run on ${bc}$ horizons taken from bag "
        f"\\texttt{{{bbag}}} and re-solved with the deployed profile, using each horizon's "
        f"own optimal input sequence instead of a constant-velocity arc. The step is varied "
        f"by subdividing the prediction interval while the input stays constant over each "
        f"$\\Delta t$, so an order remains measurable on a real trajectory: it comes out "
        f"${boe}$ and ${bom}$, unchanged. The error, however, does not. At the deployed step "
        f"the median is ${be}$~m for Euler against ${bm}$~m for the mid-point rule, a factor "
        f"${bg}$ rather than the ${gain}$ of the synthetic arc, with 95th percentiles "
        f"${bep}$~m and ${bmp}$~m. The difference is the regime: over a real horizon the "
        f"commanded yaw rate spans ${bw}$~rad/s and the per-step errors partly cancel, "
        f"while the arc accumulates them along a single turn.",
        "",
        f"One number bounds the whole discussion. Holding the velocity at its post-lag value "
        f"across the interval neglects a displacement of ${bl}$~m per horizon, which is "
        f"larger than the mid-point truncation error itself: below that level refining the "
        f"pose scheme cannot buy anything, because a different modelling choice dominates.",
        "",
    ]
    grid = [[fx(x, 4), m(sci(a)), m(sci(c)), f"{a/c:.0f}"] for x, a, c in
            zip(b["dt"], b["mediana"]["euler"], b["mediana"]["midpoint"])]
    L += table("rrrr", [r"$\Delta t$ [s]", "error, Euler [m]",
                        "error, mid-point [m]", "ratio"], grid,
               f"Median position error at the end of the horizon, over {b['cicli_usati']} "
               f"horizons re-solved from bag \\texttt{{{bbag}}}, against the exact arc on "
               f"the same velocity sequence. The first row is the deployed step; the others "
               f"subdivide the prediction interval without changing the input.",
               "res:tab:integbag")

    loop = res.get("classe3", {}).get("integratore_anello")
    if loop:
        rows = []
        for mondo, r in loop.items():
            rows.append([esc(mondo),
                         fx(r["euler"]["costo_mediano"], 1),
                         fx(r["midpoint"]["costo_mediano"], 1),
                         fx(r["delta_costo_pct"], 1)])
        spread = max(abs(r["delta_costo_pct"]) for r in loop.values())
        sp = M.add("resIntegLoopSpread", fx(spread, 1), "scarto max in anello chiuso [%]")
        segni = {r["delta_costo_pct"] > 0 for r in loop.values()}
        L += [
            "",
            f"In closed loop the choice is not visible. Running the two schemes on the same "
            f"worlds with everything else fixed moves the median cost by at most "
            f"${sp}\\%$, and "
            + ("the sign changes from world to world, which is the signature of run-to-run "
               "variation rather than of an improvement"
               if len(segni) > 1 else
               "the effect is within run-to-run variation")
            + ". Only the first input is applied and \\astar{} replans, so prediction "
              "fidelity enters the executed trajectory only indirectly.",
            "",
        ]
        L += table("lrrr", ["world", "median cost, Euler", "median cost, mid-point",
                            r"$\Delta$ [\%]"], rows,
                   "Closed-loop cost with the two integrators, same profile otherwise. "
                   "A positive $\\Delta$ means the mid-point rule is cheaper.",
                   "res:tab:integloop")
    return L


def sec_prediction(res: dict, M: Macros) -> list[str]:
    e = res.get("classe3", {}).get("errore_predizione")
    if not e:
        return []
    M.group("open-loop prediction error")
    cyc = M.add("resPredCycles", str(e["cicli_usati"]), "bag cycles used")
    off = M.add("resPredOffset", fx(e["offset_k0"], 4), "offset a k=0 [m]")
    div = M.add("resPredDivergence", fx(e["divergenza_fine_orizzonte"], 3),
                "divergence at the end of the horizon [m]")
    # The two denominators come from the measurement on the REAL horizons
    # (class 2), no longer from constants written by hand at the wrong step.
    r_eul = e["divergenza_fine_orizzonte"] / e["errore_euler_orizzonte"]
    r_mid = e["divergenza_fine_orizzonte"] / e["errore_midpoint_orizzonte"]
    ve = M.add("resPredVsEuler", f"{r_eul:.0f}", "divergenza / errore Euler")
    vm = M.add("resPredVsMidpoint", f"{r_mid:.0f}", "divergenza / errore punto medio")

    dt = res["meta"]["parametri_chiave"]["dt"]
    med, p95 = e["mediana_per_k"], e["p95_per_k"]
    rows = [[str(k), fx(k * dt, 1), fx(a, 4), fx(b, 4)]
            for k, (a, b) in enumerate(zip(med, p95))]
    return [
        r"\resSubsec{Open-loop prediction error along the horizon}",
        r"\label{res:pred}",
        "",
        r"\resNote{Feeds Report \texttt{sec:model} and \texttt{sec:mismatch}. This is the "
        r"quantity that decides whether the integrator is worth improving, and it says it "
        r"is not: read it immediately after \S\,\ref{res:disc}.}",
        "",
        f"Each MPC prediction recorded in the run was compared with the pose the robot "
        f"actually reached $k\\,\\Delta t$ later, over ${cyc}$ usable cycles. "
        f"The residual at $k=0$ is ${off}$~m and measures time alignment between the two "
        f"series, not the model. Subtracting it, the prediction diverges by "
        f"${div}$~m at the end of the horizon.",
        "",
        f"That divergence is ${ve}$ times the truncation error of forward Euler over the "
        f"same window and ${vm}$ times that of the mid-point rule "
        f"(\\S\\,\\ref{{res:disc}}). The discretisation is therefore \\emph{{not}} the "
        f"limiting term of the prediction: what the horizon loses comes from the plant, "
        f"not from the integrator, and refining the scheme would buy nothing measurable "
        f"in closed loop.",
        "",
    ] + table("rrrr",
              ["$k$", "$k\\,\\Delta t$ [s]", "median error [m]", "95th pct.\\ [m]"],
              rows,
              "Open-loop prediction error along the horizon, over "
              f"${cyc}$ cycles of the recorded run \\resBag.",
              "res:tab:pred")


def sec_nlp(res: dict, M: Macros) -> list[str]:
    d = res.get("classe1", {}).get("nlp")
    if not d:
        return []
    per_N = d["per_N"]
    Ndep = res["meta"]["parametri_chiave"]["N"]
    dep = next((r for r in per_N if r["N"] == Ndep), per_N[0])

    M.group("structure and sparsity of the NLP")
    nv = M.add("resNvar", str(dep["n_var"]), f"variabili decisionali a N={Ndep}")
    nc = M.add("resNcon", str(dep["n_con"]))
    neq = M.add("resNeq", str(dep["n_eq"]), "equality constraints")
    nin = M.add("resNineq", str(dep["n_ineq"]), "box sugli ingressi")
    npar = M.add("resNpar", str(dep["n_par"]), "parametri CasADi")
    jd = M.add("resJacDensity", fx(100 * dep["jac_density"], 2), "densita' jacobiano [%]")
    hd = M.add("resHessDensity", fx(100 * dep["hess_density"], 2), "Hessian density [%]")
    # Breakdown "as in Report.tex": state/input, dynamics/initial condition,
    # barrier terms. Guarded with .get() because a cache generated before
    # nlp_structure.structure() had them does not contain them.
    if all(k in dep for k in ("n_var_state", "n_var_input", "n_eq_dyn", "n_barrier")):
        M.add("resNvarState", str(dep["n_var_state"]), "state variables")
        M.add("resNvarInput", str(dep["n_var_input"]), "input variables")
        M.add("resNeqDyn", str(dep["n_eq_dyn"]), "dynamics equalities")
        M.add("resNBarrier", str(dep["n_barrier"]), "barrier terms in the cost")
    big = max(per_N, key=lambda r: r["N"])
    M.add("resNvarBig", str(big["n_var"]), f"variabili a N={big['N']}")
    M.add("resNbig", str(big["N"]))
    M.add("resJacDensityBig", fx(100 * big["jac_density"], 2))

    rows = []
    for r in per_N:
        mark = r["N"] == Ndep
        f = (lambda s: r"\textbf{" + s + "}") if mark else (lambda s: s)
        rows.append([f(str(r["N"])), f(str(r["n_var"])), f(str(r["n_eq"])),
                     f(str(r["n_ineq"])), f(str(r["n_par"])),
                     f(str(r["jac_nnz"])), f(fx(100 * r["jac_density"], 2)),
                     f(str(r["hess_nnz"])), f(fx(100 * r["hess_density"], 2))])
    return [
        r"\resSubsec{Size and sparsity of the nonlinear program}",
        r"\label{res:nlp}",
        "",
        r"\resNote{Feeds Report \texttt{sec:dims}. The row in bold is the "
        r"deployed configuration.}",
        "",
        f"At the deployed $N=\\resN$ the program carries ${nv}$ decision variables and "
        f"${nc}$ constraints, of which ${neq}$ are equalities (the dynamics plus the "
        f"initial condition) and ${nin}$ are simple bounds on the inputs; ${npar}$ "
        f"quantities enter as CasADi parameters, so the expression graph is built once "
        f"and only numbers are written into it between cycles.",
        "",
        f"The constraint Jacobian is ${jd}\\%$ dense and the Hessian of the Lagrangian "
        f"${hd}\\%$ (Table~\\ref{{res:tab:nlp}}). Both densities fall as $N$ grows while "
        f"the nonzero counts grow linearly, which is the signature of the "
        f"multiple-shooting parametrisation: no constraint couples distant stages, so "
        f"the cost of one interior-point iteration is linear in the horizon. "
        f"\\S\\,\\ref{{res:shoot}} builds the same program in the condensed "
        f"parametrisation and measures what that trade actually is, rather than "
        f"asserting it.",
        "",
    ] + table("rrrrrrrrr",
              ["$N$", "vars", "eq.", "bounds", "params",
               "jac nnz", "jac dens.\\ [\\%]", "hess nnz", "hess dens.\\ [\\%]"],
              rows,
              "Size and sparsity of the NLP against the prediction horizon, from the "
              "deployed profile. Bold: the deployed configuration.",
              "res:tab:nlp")


def sec_derivatives(res: dict, M: Macros) -> list[str]:
    d = res.get("classe1", {}).get("derivate")
    if not d:
        return []
    M.group("derivatives: AD against finite differences")
    nvar = M.add("resDerivNvar", str(d["n_variabili"]), "variables of the test point")
    tf = M.add("resTimeF", fx(d["t_f_us"], 1), "one evaluation of f [us]")
    tg = M.add("resTimeGrad", fx(d["t_grad_ad_us"], 1), "one gradient by AD [us]")
    rat = M.add("resADratio", fx(d["ad_in_valutazioni_di_f"], 2), "AD in evaluations of f")
    lo = M.add("resADratioMin", fx(d["ad_in_valutazioni_di_f_min"], 2))
    hi = M.add("resADratioMax", fx(d["ad_in_valutazioni_di_f_max"], 2))
    nf = M.add("resFDforwardEvals", str(d["valutazioni_fd_avanti"]))
    ncq = M.add("resFDcentralEvals", str(d["valutazioni_fd_centrate"]))
    ef = M.add("resFDforwardErr", sci(d["miglior_err_avanti"]))
    ecq = M.add("resFDcentralErr", sci(d["miglior_err_centrate"]))
    bud = M.add("resCycleBudget", fx(d["budget_ciclo_ms"], 0), "cycle budget [ms]")
    shr = M.add("resFDcentralBudget", pc(d["quota_budget_fd_centrate"], 0),
                "share of the budget for central FD [%]")

    rows = [[m(sci(r["h"])), m(sci(r["err_avanti"])), m(sci(r["err_centrate"]))]
            for r in d["tabella_passi"]]
    L = [
        r"\resSubsec{Derivatives: algorithmic differentiation against finite differences}",
        r"\label{res:ad}",
        "",
        r"\resNote{Feeds Report \texttt{sec:solver} / \texttt{sec:impl}, which currently "
        r"assert that ``exact derivatives are available from CasADi's AD'' without "
        r"measuring the alternative.}",
        "",
        f"At a representative solve the objective has ${nvar}$ variables. One evaluation "
        f"of the objective costs ${tf}~\\mu$s and one full gradient by reverse-mode "
        f"algorithmic differentiation ${tg}~\\mu$s, i.e.\\ ${rat}$ objective evaluations "
        f"(median of repeated pairs, range ${lo}$--${hi}$), independently of the number "
        f"of variables. The same gradient by finite differences costs ${nf}$ evaluations "
        f"forward and ${ncq}$ central, and is less accurate at every step size: the "
        f"best relative error reachable is ${ef}$ forward and ${ecq}$ central "
        f"(Table~\\ref{{res:tab:fd}}), against machine precision for AD.",
        "",
        f"The cost is the argument that closes the question for a real-time loop: central "
        f"differences alone would consume ${shr}\\%$ of the ${bud}$~ms cycle budget, for "
        f"a gradient that is worse. The ratio itself is a micro-benchmark on "
        f"$\\sim100~\\mu$s timings and its second digit is not meaningful; what is stable, "
        f"and is the point, is that it stays a small constant.",
        "",
    ]
    if not d.get("ad_ratio_attendibile", True):
        L += [r"\resNote{The AD/objective ratio came out below $1$ in at least one "
              r"repetition, which is physically impossible: re-run on an idle machine "
              r"before quoting it.}", ""]
    L += table("rrr", ["step $h$", "rel.\\ error, forward", "rel.\\ error, central"],
               rows,
               "Accuracy of the finite-difference gradient against the step size, "
               "measured against the AD gradient. The optimum sits at "
               f"$h={sci(d['h_ottimo_avanti'])}$ forward (theory: "
               f"$\\sqrt{{\\varepsilon}}={sci(d['h_teorico_avanti'])}$) and "
               f"$h={sci(d['h_ottimo_centrate'])}$ central (theory: "
               f"$\\varepsilon^{{1/3}}={sci(d['h_teorico_centrate'])}$).",
               "res:tab:fd")
    return L


def sec_hessian(res: dict, M: Macros) -> list[str]:
    d = res.get("classe1", {}).get("hessiana")
    if not d:
        return []
    ex, lb = d.get("exact"), d.get("limited-memory")
    if not ex or not lb:
        return []
    M.group("exact Hessian against L-BFGS")
    ie = M.add("resIterExactHess", str(ex["iterazioni"]))
    il = M.add("resIterLBFGS", str(lb["iterazioni"]))
    sav = M.add("resHessIterSaving", pc(1 - ex["iterazioni"] / lb["iterazioni"], 0),
                "iterations saved [%]")
    # the best iteration count in bold: it is the comparison the table exists to
    # make, and the guidelines ask for it to be highlighted
    def _it(v, best):
        return r"\textbf{" + str(v) + "}" if v == best else str(v)
    best_it = min(ex["iterazioni"], lb["iterazioni"])
    rows = [["exact Hessian", _it(ex["iterazioni"], best_it), fx(ex["solve_ms"], 1),
             m(smart(ex["J"])), tt(ex["status"])],
            ["L-BFGS", _it(lb["iterazioni"], best_it), fx(lb["solve_ms"], 1),
             m(smart(lb["J"])), tt(lb["status"])]]
    return [
        r"\resSubsec{Exact Hessian against a quasi-Newton approximation}",
        r"\label{res:hess}",
        "",
        r"\resNote{Feeds Report \texttt{sec:solver}: it is the second half of the AD "
        r"argument --- AD is what makes the exact Hessian affordable, and this is what "
        r"the exact Hessian buys.}",
        "",
        f"Solving the same instance with the exact Hessian of the Lagrangian takes "
        f"${ie}$ iterations against ${il}$ with the limited-memory quasi-Newton "
        f"approximation, a ${sav}\\%$ reduction, and both converge to the same objective "
        f"value (Table~\\ref{{res:tab:hess}}). Since CasADi supplies the exact second "
        f"derivatives at a cost comparable to the first, the full Newton step is the "
        f"cheaper option here, not the more expensive one.",
        "",
    ] + table("lrrrl",
              ["Hessian", "iterations", "solve [ms]", "$J^\\star$", "status"], rows,
              "Interior-point iterations with the exact Hessian and with the "
              "limited-memory quasi-Newton approximation, on the same instance. "
              "Bold: the lower iteration count.",
              "res:tab:hess")


def sec_kkt(res: dict, M: Macros) -> list[str]:
    d = res.get("classe2", {}).get("kkt")
    if not d or not d.get("profilo"):
        return []
    prof = d["profilo"]
    M.group("optimality conditions along the mission")
    nc = M.add("resKKTcycles", str(len(prof)), "cicli analizzati")
    li = M.add("resLICQalways", yesno(d["licq_sempre"]))
    st = M.add("resStrictAlways", yesno(d["complementarita_stretta_sempre"]))
    so = M.add("resSOCalways", yesno(d["soc_c2_sempre"]))
    cmin = M.add("resConeMin", str(d["cono_critico_min"]))
    cmax = M.add("resConeMax", str(d["cono_critico_max"]))
    # Along the mission the cone narrows: that is the fact to report, and its
    # direction has to be read from the data instead of assumed.
    cone_first = prof[0]["dim_cono_critico"]
    cone_last = prof[-1]["dim_cono_critico"]
    M.add("resConeFirst", str(cone_first), "cono al primo ciclo campionato")
    M.add("resConeLast", str(cone_last), "cono all'ultimo")
    act_last = prof[-1]["n_attivi_totali"]
    M.add("resActiveLast", str(act_last), "vincoli attivi all'ultimo ciclo")
    if cone_last < cone_first:
        cone_txt = (
            f"The quantity worth reporting is what that does to the critical cone, whose "
            f"dimension falls from $\\resConeFirst$ at the first sampled cycle to "
            f"$\\resConeLast$ at the last: with $\\resNvar$ variables and "
            f"$\\resActiveLast$ active constraints, only $\\resConeLast$ degree(s) of "
            f"freedom remain. Over the mission the controller moves from "
            f"\\emph{{cost-driven}} to \\emph{{constraint-driven}}: towards the end it "
            f"no longer chooses the trajectory so much as undergo it. The input envelope "
            f"is what does this --- a lateral bound of $\\resVyMax$~m/s does not "
            f"\\emph{{limit}} a degree of freedom, it \\emph{{removes}} it. The same "
            f"asymmetry bounds what any state constraint can ask of this robot: it can "
            f"advance along its own heading, but it cannot translate away from a wall, "
            f"which is the limit \\S\\,\\ref{{res:robust}} runs into from the other "
            f"side.")
    else:
        cone_txt = (
            f"The critical cone does not contract along this profile "
            f"($\\resConeFirst$ at the first sampled cycle, $\\resConeLast$ at the "
            f"last), so on this run the controller stays cost-driven throughout. The "
            f"cone dimension is worth watching precisely because it need not: it is the "
            f"number of directions the optimizer still has left after the active "
            f"constraints have taken their share.")
    lmin = min(p["hess_proj_lambda_min"] for p in prof)
    lm = M.add("resLambdaMinWorst", sci(lmin), "minimum over the profile of projected lambda_min")
    nbmax = max(p.get("n_attivi_ineq", 0) for p in prof)
    M.add("resActiveBoundsMax", str(nbmax), "maximum number of active box constraints")
    gl = max(p.get("grad_L_inf", 0.0) for p in prof)
    M.add("resGradLagWorst", sci(gl), "worst stationarity residual")

    rows = [[str(p["ciclo"]), fx(p["t"], 0), str(p["n_attivi_totali"]),
             str(p.get("n_attivi_ineq", 0)), str(p["rango_jacobiano_attivo"]),
             yesno(p["licq"]), str(p["dim_cono_critico"]),
             m(sci(p["hess_proj_lambda_min"]))] for p in prof]
    return [
        r"\resSubsec{Optimality conditions along the mission}",
        r"\label{res:kkt}",
        "",
        r"\resNote{New material: the report has no KKT section at all. It belongs next to "
        r"\texttt{sec:constraints}, and it is what licenses calling $z^\star$ an optimum "
        r"rather than ``what IPOPT returned''.}",
        "",
        f"The first- and second-order conditions were checked at ${nc}$ cycles sampled "
        f"along a recorded mission. LICQ holds at every one of them (${li}$), strict "
        f"complementarity holds at every one (${st}$), and the reduced Hessian is "
        f"positive definite on the critical cone at every one (${so}$), the smallest "
        f"projected eigenvalue over the profile being ${lm}$. The solution is therefore a "
        f"strict local minimum satisfying the second-order sufficient conditions, and the "
        f"multipliers are unique.",
        "",
        f"The structural remark first: the obstacle terms live in the objective, not in "
        f"the constraints, so there is no obstacle multiplier and no non-trivial "
        f"active-set combinatorics to resolve. What remains active is the dynamics plus "
        f"the input bounds, and the bounds are active up to $\\resActiveBoundsMax$ times "
        f"per cycle.",
        "",
        cone_txt,
        "",
        f"The cone dimension over the sampled profile ranges between ${cmin}$ and "
        f"${cmax}$ (Table~\\ref{{res:tab:kkt}}), and the identity "
        f"$\\dim(\\text{{cone}}) = n_{{\\text{{var}}}} - \\#\\text{{active}}$ holds "
        f"at every cycle, which is what strict complementarity buys.",
        "",
    ] + table("rrrrrcrr",
              ["cycle", "$t$ [s]", "active", "of which bounds", "rank",
               "LICQ", "cone dim.", "$\\lambda_{\\min}$ proj."],
              rows,
              "Optimality diagnostics at sampled cycles of the recorded run \\resBag. "
              "``active'' counts equalities and active bounds together; ``rank'' is the "
              "rank of the Jacobian of the active constraints, so LICQ holds when the "
              "two coincide.",
              "res:tab:kkt")


def sec_penalty(res: dict, M: Macros) -> list[str]:
    d = res.get("classe1", {}).get("penalita_esatta")
    if not d:
        return []
    M.group("exact l1 penalty against l2")
    ds = M.add("resDsafe", fx(d["d_safe"], 2), "imposed safety distance [m]")
    mu = M.add("resMuMax", smart(d["max_mu_vincolo_distanza"]),
               "largest multiplier of the distance constraint")
    rz = d.get("rho_slack_l1_nullo")
    rzs = M.add("resRhoLoneZero", sci(rz, 0) if rz else DASH, "rho a cui lo slack l1 si annulla")
    slope = d.get("pendenza_l2_coda")
    slope_ok = slope is not None and not math.isnan(slope)
    sl = M.add("resSlopeLtwo", fx(slope, 2) if slope_ok else DASH,
               "log-log slope of the l2 tail")
    # The 1/rho decay is the prediction; if the measurement does not match it, say
    # so, instead of writing "as predicted" next to a number that is not.
    if slope_ok and abs(slope + 1.0) <= 0.15:
        slope_txt = (f"the fitted log--log slope of its tail being ${sl}$ against a "
                     f"predicted $-1$")
    else:
        slope_txt = (f"but the fitted log--log slope of its tail, ${sl}$, does not yet "
                     f"reach the predicted $-1$: the fit uses only the last three "
                     f"points of the weight sweep, so it needs the full grid rather "
                     f"than the \\texttt{{-{'-'}quick}} one before it can be quoted")

    rows = []
    for r in d["tabella"]:
        # the exact zero is THE result of the table (exact-penalty theorem)
        s1 = r"$\mathbf{0}$" if r["slack_l1"] < 1e-8 else f"${sci(r['slack_l1'])}$"
        rows.append([f"${sci(r['rho'], 0)}$", s1, f"${sci(r['slack_l2'])}$"])
    return [
        r"\resSubsec{Soft obstacle constraint: exact $\ell^1$ penalty against $\ell^2$}",
        r"\label{res:penalty}",
        "",
        r"\resNote{New material, and the strongest single addition available to the "
        r"report: it turns \texttt{sec:barrier}'s admission that ``the barrier is not an "
        r"exact penalty, so a finite weight admits a finite violation'' from a caveat "
        r"into a measurement, with the threshold at which the violation is exactly zero.}",
        "",
        f"The obstacle term was re-posed as a genuine inequality constraint "
        f"$d(x_k,\\mathcal{{P}})\\ge d_{{\\mathrm{{safe}}}}$ with $d_{{\\mathrm{{safe}}}}"
        f"={ds}$~m, relaxed by a slack variable penalised either linearly ($\\ell^1$) or "
        f"quadratically ($\\ell^2$) with weight $\\rho_s$. Solved as a hard constraint the "
        f"instance is feasible and the largest multiplier of an active distance "
        f"constraint is $\\mu^\\star={mu}$.",
        "",
        f"The two penalties then behave exactly as the exact-penalty theorem predicts "
        f"(Table~\\ref{{res:tab:penalty}}). The $\\ell^1$ slack is a threshold "
        f"phenomenon: it is nonzero below the threshold and drops to zero --- not small, "
        f"zero to solver tolerance --- from $\\rho_s={rzs}$ onwards, i.e.\\ once $\\rho_s$ "
        f"exceeds the multiplier of the corresponding hard constraint. The $\\ell^2$ "
        f"slack instead decays like $1/\\rho_s$, {slope_txt}, and never reaches zero at "
        f"any finite weight.",
        "",
        f"The practical reading for this stack is that a soft constraint can be made "
        f"\\emph{{exactly}} satisfied at a finite, computable weight, provided the "
        f"penalty is non-smooth at the origin; the smooth quadratic relaxation that is "
        f"more comfortable for the solver is precisely the one that can never close the "
        f"violation.",
        "",
    ] + table("rrr",
              [r"$\rho_s$", r"max slack, $\ell^1$ [m]", r"max slack, $\ell^2$ [m]"],
              rows,
              "Residual constraint violation against the penalty weight, for the "
              "non-smooth and the smooth relaxation of the same distance constraint. "
              "Bold: the violations that are exactly zero, which is the result the "
              "exact-penalty theorem predicts and the smooth relaxation never attains. "
              "Zero entries are below the solver tolerance of "
              "$1\\times 10^{-8}$~m.",
              "res:tab:penalty")


def sec_terminal(res: dict, M: Macros) -> list[str]:
    d = res.get("classe3", {}).get("vincolo_terminale")
    if not d:
        return []
    M.group("terminal equilibrium constraint")
    sm = M.add("resTermSlackMax", sci(d["slack_max"]), "slack terminale massimo")
    fe = M.add("resTermFeasible", yesno(d["sempre_ammissibile"]))
    lo = M.add("resTermCostMin", pc(d["costo_relativo_min"], 1), "minimum cost [%]")
    hi = M.add("resTermCostMax", pc(d["costo_relativo_max"], 1), "maximum cost [%]")
    # Absolute figure: it is the one that can be quoted without misleading. See
    # the comment in make_results._classe3. Guarded with .get() because a cache
    # generated before this addition does not contain it.
    has_abs = all(k in d for k in ("costo_assoluto_min", "costo_assoluto_max",
                                   "J_min", "J_max"))
    if has_abs:
        am = M.add("resTermAbsMin", fx(d["costo_assoluto_min"], 1),
                   "smallest absolute increase of J*")
        ax = M.add("resTermAbsMax", fx(d["costo_assoluto_max"], 1),
                   "largest absolute increase of J*")
        jl = M.add("resTermJMin", fx(d["J_min"], 1), "smallest J* across the cycles")
        jh = M.add("resTermJMax", fx(d["J_max"], 1), "largest J* across the cycles")
    # The discrete lag is 1 - exp(-dt/tau): with tau << dt it is 1, i.e. v(k+1)=u(k).
    tau = float(M.params.get("tau_v", 0.0)) or 1e-12
    dtv = float(M.params.get("dt", 0.0))
    lag = 1.0 - math.exp(-dtv / tau) if dtv else float("nan")
    M.add("resLagDiscrete", fx(lag, 6), "1 - exp(-dt/tau) al profilo deployato")
    degenerate = lag > 0.999
    L = [
        r"\resSubsec{Adding a terminal equilibrium constraint}",
        r"\label{res:terminal}",
        "",
        r"\resNote{Feeds Report \texttt{sec:terminal}, which states that the formulation "
        r"carries no terminal ingredient and that none of the standard guarantees apply. "
        r"This measures what supplying one would actually cost.}",
        "",
        f"The program was re-solved with a terminal equilibrium constraint --- the "
        f"velocity states driven to zero at the last node, relaxed by a slack so that the "
        f"comparison can never be decided by infeasibility. Across the sampled cycles the "
        f"terminal slack never leaves zero (maximum ${sm}$, always admissible: ${fe}$): "
        f"the constraint is reachable at every operating point tested, so the terminal "
        f"set is not empty in practice and recursive feasibility is not obtained at the "
        f"price of an infeasible program.",
        "",
        (f"The cost of imposing it is best read in absolute terms: the optimal objective "
         f"rises by between ${am}$ and ${ax}$ across the sampled cycles, whose objectives "
         f"themselves range from ${jl}$ to ${jh}$. The same increases therefore read as "
         f"anything between ${lo}\\%$ and ${hi}\\%$, which is a statement about the "
         f"denominator rather than about the constraint. The plan has to predict its own "
         f"braking, and the braking is discarded at the next cycle."
         if has_abs else
         f"The cost of imposing it, measured as the relative increase of the optimal "
         f"objective, ranges from ${lo}\\%$ to ${hi}\\%$ depending on the cycle."),
        "",
    ]
    if degenerate:
        L += [
            f"That the slack is always zero is not by itself evidence that the constraint "
            f"works --- an unimplemented constraint would report the same thing --- and "
            f"the reason it is zero has to be stated, because it is a property of this "
            f"profile and not of the method. The discrete lag is "
            f"$1-e^{{-\\Delta t/\\tau}}=\\resLagDiscrete$ at $\\tau=\\resTauV$~s "
            f"against $\\Delta t=\\resDt$~s, i.e.\\ degenerate: the model reduces to "
            f"$v_{{k+1}}=u_k$ and the robot reaches zero velocity in a single step, so the "
            f"terminal set is trivially reachable from anywhere. On hardware, with an "
            f"identified actuator time constant, the slack becomes the quantity that "
            f"decides the maximum speed from which the horizon can still bring the robot "
            f"to a stop --- which is the physical reading of the terminal feasible set, "
            f"and the form in which this constraint would actually bind.",
            "",
        ]
    return L


def sec_bifurcation(res: dict, M: Macros) -> list[str]:
    d = res.get("classe2", {}).get("biforcazione")
    if not d:
        return []
    cp = d.get("centred_pillar")
    if not cp:
        return []
    M.group("regularity of the solution and bifurcation")
    lo = cp.get("soglia_inf")
    hi = cp.get("soglia_sup")
    lo_m = M.add("resBifLow", smart(lo) if lo is not None else DASH,
                 "ultimo W senza biforcazione")
    hi_m = M.add("resBifHigh", smart(hi) if hi is not None else DASH,
                 "first W with a bifurcation")
    below = M.add("resBifDeployedBelow", yesno(cp.get("deployato_sotto_soglia")))
    bb = d.get("bag_ciclo_piu_impegnativo", {})
    bc = M.add("resBifBagCycle", str(bb.get("ciclo", "")) or DASH, "most demanding cycle")
    be = M.add("resBifBagEver", yesno(bb.get("biforca_mai")))
    # When it bifurcates, the two minima are not equivalent: the cost gap is the
    # interesting datum, because it says that the warm start does not only choose
    # finisce ma QUANTO si paga.
    split = [r for r in cp["tabella"] if r["sep"] > 1e-3]
    gap = max((abs(r["JL"] - r["JR"]) / max(abs(r["JL"]), 1e-9) for r in split),
              default=0.0)
    M.add("resBifCostGap", pc(gap, 2) if split else DASH,
          "relative gap between the two minima [%]")

    rows = [[m(smart(r["W"])), m(sci(r["sep"])), m(smart(r["JL"])),
             m(smart(r["JR"])), str(r["itL"]), str(r["itR"])]
            for r in cp["tabella"]]
    return [
        r"\resSubsec{Regularity of the solution: the left--right bifurcation}",
        r"\label{res:bif}",
        "",
        r"\resNote{Feeds Report \texttt{sec:barriersweep}. The report already shows that "
        r"the barrier weight governs a cliff; this gives the cliff its mechanism, and it "
        r"is the one place where the non-convexity of the program becomes visible as a "
        r"discontinuity of the control law rather than as an iteration count.}",
        "",
        f"With an obstacle centred on the reference, the program admits two symmetric "
        f"local minima --- pass left, pass right --- and the solution as a function of "
        f"the barrier weight is regular only as long as one of them dominates. Solving "
        f"from a left-biased and a right-biased initial guess and measuring the "
        f"separation between the two returned trajectories locates the transition "
        f"between $\\Wobs={lo_m}$ and $\\Wobs={hi_m}$ (Table~\\ref{{res:tab:bif}}). Below "
        f"it the two guesses collapse onto the same trajectory; above it they do not, and "
        f"the optimizer's choice becomes a discontinuous function of the current state.",
        "",
        f"The deployed weight sits below the threshold (${below}$), and on the hardest "
        f"cycle of the recorded run --- cycle ${bc}$, the one with the most obstacles "
        f"inside the search radius --- the sweep never bifurcates (${be}$). The design is "
        f"therefore on the regular side of the transition rather than accidentally past "
        f"it, which is the statement the report needs in order to treat the MPC law as "
        f"well defined.",
        "",
        f"Two readings follow, and both are actionable. First, past the threshold the two "
        f"minima are \\emph{{not}} equivalent: their objective values differ by up to "
        f"$\\resBifCostGap\\%$, so the initial guess decides not merely which side the "
        f"robot passes on but how much the manoeuvre costs. Second, since the deployed "
        f"weights are on the regular side, the cost-spike guard that clears the warm-start "
        f"cache when the objective jumps is protecting against a phenomenon that does not "
        f"occur at these weights --- worth stating as a measured fact about the deployed "
        f"configuration rather than removing on the strength of one sweep.",
        "",
    ] + table("rrrrrr",
              [r"$\Wobs$", "separation [m]", "$J^\\star$ left", "$J^\\star$ right",
               "iter.\\ left", "iter.\\ right"], rows,
              "Left-biased against right-biased solve of the same instance, against the "
              "barrier weight. The separation is the distance between the two returned "
              "trajectories: a value at solver tolerance means the two guesses converged "
              "to the same minimum.",
              "res:tab:bif")


def sec_pathfollowing(res: dict, M: Macros) -> list[str]:
    d = res.get("classe3", {}).get("path_following")
    if not d:
        return []
    M.group("path following in theta against time-based reference")
    vt = M.add("resVxTime", fx(d["vx_media_time"], 3), "vx media, riferimento a tempo")
    vh = M.add("resVxTheta", fx(d["vx_media_theta"], 3), "vx media, ascissa")
    at = M.add("resAdvanceTime", fx(d["spostamento_time"], 3), "avanzamento [m]")
    ah = M.add("resAdvanceTheta", fx(d["spostamento_theta"], 3))
    ag = M.add("resAdvanceGain", pc(d["guadagno_spostamento"], 0), "guadagno [%]")
    it = M.add("resIterTime", fx(d["iter_time"], 1))
    ih = M.add("resIterTheta", fx(d["iter_theta"], 1))
    un = M.add("resVrefUnused", pc(d["velocita_inutilizzata_da_v_ref"], 0),
               "velocita' lasciata inutilizzata da v_ref [%]")
    nc = M.add("resPFcycles", str(len(d["cicli"])), "cicli confrontati")

    rows = [["mean $v_x$ [m/s]", vt.join(["$", "$"]), vh.join(["$", "$"])],
            ["advance over the horizon [m]", at.join(["$", "$"]), ah.join(["$", "$"])],
            ["IPOPT iterations", it.join(["$", "$"]), ih.join(["$", "$"])]]
    return [
        r"\resSubsec{Reference generation: time-parametrised against path-parametrised}",
        r"\label{res:pf}",
        "",
        r"\resNote{Feeds Report \texttt{sec:refwarm}, which defines the reference by "
        r"advancing along the path at a fixed cruise speed. This measures what that fixed "
        r"speed costs.}",
        "",
        f"The deployed reference samples the smoothed path at a constant cruise speed "
        f"$v_{{\\mathrm{{ref}}}}=\\resVref$~m/s, which is set below $v_{{x,\\max}}$ on "
        f"purpose --- with $v_{{\\mathrm{{ref}}}}=v_{{x,\\max}}$ the tracker saturates "
        f"permanently and never settles onto the path. The price is that ${un}\\%$ of the "
        f"available forward speed is unreachable by construction: no cost weight can "
        f"recover it, because the reference itself never asks for it.",
        "",
        f"Re-posing the same program with the path abscissa as a decision variable --- "
        f"the tracker chooses how far to advance rather than being told --- removes the "
        f"constant and lets the bound do the limiting. Over ${nc}$ cycles replayed from "
        f"the recorded run, the mean commanded speed rises from ${vt}$ to ${vh}$~m/s and "
        f"the distance covered within one horizon from ${at}$ to ${ah}$~m, a ${ag}\\%$ "
        f"gain (Table~\\ref{{res:tab:pf}}).",
        "",
        f"It is not free: the iteration count rises from ${it}$ to ${ih}$, because the "
        f"extra decision variable removes the term that was pinning the solution along "
        f"the path. Reported as a trade rather than as an improvement, this is the "
        f"cleanest reformulation available to the stack, and the one that would let "
        f"$v_{{\\mathrm{{ref}}}}$ disappear from the parameter file.",
        "",
    ] + table("lrr",
              ["quantity", "time-parametrised", "path-parametrised"], rows,
              f"Time-parametrised against path-parametrised reference, averaged over "
              f"${nc}$ cycles replayed from the recorded run \\resBag.",
              "res:tab:pf")


def sec_horizon(extra: dict, M: Macros) -> list[str]:
    d = extra.get("horizon")
    if not d:
        return []
    try:
        rows_in = d["righe"]
        budget = float(d["budget_ms"])
        depN, depdt = int(d["deployato"]["N"]), float(d["deployato"]["dt"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  [horizon_sweep.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []

    M.group("horizon campaign (N, dt)")
    M.add("resHorizonBudget", fx(budget, 0), "cycle budget [ms]")
    over = [r for r in rows_in if r.get("solve_ms_p95", 0) > budget]
    M.add("resHorizonOverBudget", str(len(over)), "configurazioni oltre budget")
    M.add("resHorizonConfigs", str(len(rows_in)), "configurazioni provate")
    dep_rows = [r for r in rows_in if r["N"] == depN and abs(r["dt"] - depdt) < 1e-9]
    if dep_rows:
        M.add("resHorizonDepTail", fx(max(r["solve_ms_p95"] for r in dep_rows), 1),
              "worst p95 of the deployed configuration [ms]")

    scenarios: dict[str, list] = {}
    for r in rows_in:
        scenarios.setdefault(str(r.get("scenario", "default")), []).append(r)

    # Aggregation by (N, dt) across the scenarios, conservatively: the worst case
    # is taken for the clearance and the mean for the times,
    # because averaging the clearance would hide the near-collision.
    agg: dict[tuple, dict] = {}
    for r in rows_in:
        k = (int(r["N"]), round(float(r["dt"]), 4))
        a = agg.setdefault(k, {"t": [], "c": [], "p": [], "goal": True})
        # A configuration that does NOT reach the goal has no time to goal: the
        # field is null. It has to be excluded from the mean instead of being
        # converted, and the configuration marked as failed — which is already how
        # it is kept out of the non-dominance ranking.
        t = r.get("tempo_al_goal_s")
        if t is not None:
            a["t"].append(float(t))
        for campo, chiave in (("c", "clearance_min"), ("p", "solve_ms_p95")):
            v = r.get(chiave)
            if v is not None:
                a[campo].append(float(v))
        a["goal"] = a["goal"] and bool(r.get("goal_raggiunto"))
    pts = []
    for (N, dtv), a in agg.items():
        if not a["t"] or not a["c"] or not a["p"]:
            a["goal"] = False
        pts.append({"N": N, "dt": dtv, "T": N * dtv, "goal": a["goal"],
                    "t": sum(a["t"]) / len(a["t"]) if a["t"] else float("inf"),
                    "c": min(a["c"]) if a["c"] else 0.0,
                    "p": max(a["p"]) if a["p"] else float("inf")})

    def dominates(x, y):
        return (x["t"] <= y["t"] and x["c"] >= y["c"] and x["p"] <= y["p"]
                and (x["t"] < y["t"] or x["c"] > y["c"] or x["p"] < y["p"]))

    ok = [q for q in pts if q["goal"]]
    nondom = [q for q in ok if not any(dominates(o, q) for o in ok if o is not q)]
    dep = next((q for q in pts if q["N"] == depN and abs(q["dt"] - depdt) < 1e-9), None)
    dep_nd = dep is not None and dep in nondom

    SPLIT = 6.0
    lowb = [q for q in ok if q["T"] < SPLIT]
    highb = [q for q in ok if q["T"] >= SPLIT]
    banded = bool(lowb and highb)

    M.add("resHorizonNondom", str(len(nondom)), "non-dominated configurations")
    M.add("resHorizonDepNondom", yesno(dep_nd) if dep else DASH)
    if banded:
        M.add("resHorizonSplit", fx(SPLIT, 0), "threshold of the band [s]")
        M.add("resHorizonLowT", fx(sum(q["t"] for q in lowb) / len(lowb), 1))
        M.add("resHorizonLowC", fx(min(q["c"] for q in lowb), 3))
        M.add("resHorizonHighT", fx(sum(q["t"] for q in highb) / len(highb), 1))
        M.add("resHorizonHighC", fx(min(q["c"] for q in highb), 3))
    cheap = min(nondom, key=lambda q: q["p"]) if nondom else None
    if cheap:
        M.add("resHorizonCheapN", str(cheap["N"]))
        M.add("resHorizonCheapDt", fx(cheap["dt"], 2))
        M.add("resHorizonCheapPtail", fx(cheap["p"], 1), "p95 of the cheapest one [ms]")

    L = [
        r"\resSubsec{Prediction horizon and sampling time}",
        r"\label{res:horizon}",
        "",
        r"\resNote{Feeds Report \texttt{sec:horizon}. Note that here $N$ and $\Delta t$ "
        r"are swept independently rather than at a fixed look-ahead, so the sweep "
        r"collapses into a single table.}",
        "",
        f"The two parameters are not interchangeable: the product $N\\Delta t$ sets how "
        f"far the controller sees, $N$ alone sets how much the solve costs, and "
        f"$\\Delta t$ alone sets how faithful the prediction is. They were therefore "
        f"swept jointly over $\\resHorizonConfigs$ configurations, in closed loop, "
        f"against a cycle budget of $\\resHorizonBudget$~ms; "
        f"$\\resHorizonOverBudget$ of them exceed it at the 95th percentile of the "
        f"per-cycle solve time.",
        "",
    ]
    if banded:
        t_lo = sum(q["t"] for q in lowb) / len(lowb)
        t_hi = sum(q["t"] for q in highb) / len(highb)
        c_lo, c_hi = min(q["c"] for q in lowb), min(q["c"] for q in highb)
        worse = t_hi > t_lo
        # Saying "worse on both fronts" when the clearance improves is
        # false: they are two axes, and have to be read separately. And if the
        # clearance is zero in both bands, that axis does not discriminate at all
        # and passing it off as an advantage would be worse than staying silent.
        both_graze = c_lo < 0.02 and c_hi < 0.02
        if both_graze:
            coda = ("--- and the clearance axis does not discriminate: both groups "
                    "come within a couple of centimetres of an obstacle, so on these "
                    "scenarios neither horizon is being solved safely and only the "
                    "time is informative")
        elif c_hi > c_lo:
            coda = ("--- so it is a trade and not a dominance: the longer horizon buys "
                    "clearance and pays for it in time")
        else:
            coda = "--- worse on both counts, not a trade"
        if worse:
            L += [
                f"The headline is counter-intuitive and worth stating first: "
                f"\\emph{{lengthening the horizon makes the closed loop worse}}. Split "
                f"at $N\\Delta t=\\resHorizonSplit$~s, the short-horizon group reaches "
                f"the goal in $\\resHorizonLowT$~s with a worst-case clearance of "
                f"$\\resHorizonLowC$~m, against $\\resHorizonHighT$~s and "
                f"$\\resHorizonHighC$~m for the long-horizon group {coda}. "
                f"The mechanism is the same one that produces the "
                f"livelock elsewhere in the stack: the reference extends over a path the "
                f"discrete planner will replan anyway, so a longer horizon commits the "
                f"controller to tracking a target that is already due to change. "
                f"That explanation has a competitor --- simply too many decision "
                f"variables --- and \\S\\,\\ref{{res:nc}} is the experiment that "
                f"separates the two; it reports there which one this data supports.",
                "",
            ]
        else:
            L += [
                f"Grouped at $N\\Delta t=\\resHorizonSplit$~s, the two bands reach the "
                f"goal in $\\resHorizonLowT$~s and $\\resHorizonHighT$~s with "
                f"worst-case clearances of $\\resHorizonLowC$~m and "
                f"$\\resHorizonHighC$~m: on this run a longer horizon does not degrade "
                f"the closed loop, which is worth recording because it did on earlier "
                f"sweeps and the effect is scenario-dependent.",
                "",
            ]
    if nondom and dep is not None:
        verdict = ("is itself non-dominated" if dep_nd else
                   "is \\emph{dominated}: another configuration matches or beats it on "
                   "all three")
        cheaper = ""
        if cheap and not dep_nd:
            cheaper = (f" The cheapest point of the non-dominated set is "
                       f"$N=\\resHorizonCheapN$, $\\Delta t=\\resHorizonCheapDt$~s, "
                       f"at $\\resHorizonCheapPtail$~ms of 95th-percentile solve time "
                       f"against $\\resHorizonDepTail$~ms for the deployed one.")
        L += [
            f"Ranking the aggregated configurations on the three objectives that matter "
            f"---time to goal, worst-case clearance and tail solve time--- leaves "
            f"$\\resHorizonNondom$ non-dominated points, and the deployed "
            f"$N=\\resN$, $\\Delta t=\\resDt$~s {verdict}.{cheaper}",
            "",
            r"\resNote{\textbf{Before changing the deployed profile.} These are "
            r"synthetic scenarios with static obstacles and frequent replanning. A very "
            r"short horizon holds up only because the discrete planner is doing the "
            r"avoidance; with dynamic obstacles, or a slower planner, the margin would "
            r"disappear. The result is a reason to run the comparison on real missions, "
            r"not a reason to retune from a table.}",
            "",
        ]
    for name, rows_s in sorted(scenarios.items()):
        body = []
        for r in sorted(rows_s, key=lambda r: (r["N"], r["dt"])):
            mark = r["N"] == depN and abs(r["dt"] - depdt) < 1e-9
            f = (lambda s: r"\textbf{" + s + "}") if mark else (lambda s: s)
            body.append([
                f(str(r["N"])), f(fx(r["dt"], 2)), f(fx(r["T_orizzonte"], 1)),
                f(str(r["n_var"])), f(fx(r["solve_ms_mediana"], 1)),
                f(fx(r["solve_ms_p95"], 1)), f(yesno(r["goal_raggiunto"])),
                f(fx(r["tempo_al_goal_s"], 1)), f(fx(r["clearance_min"], 3)),
            ])
        L += table("rrrrrrcrr",
                   ["$N$", r"$\Delta t$ [s]", "$T$ [s]", "vars", "median [ms]",
                    "p95 [ms]", "goal", "TTG [s]", "min clear.\\ [m]"], body,
                   f"Horizon campaign, scenario \\texttt{{{esc(name)}}}. Bold: the "
                   f"deployed configuration $N={depN}$, $\\Delta t={fx(depdt,2)}$~s.",
                   f"res:tab:horizon:{re.sub('[^a-z0-9]', '', name.lower())}")
    return L


def sec_pareto(extra: dict, M: Macros) -> list[str]:
    d = extra.get("pareto")
    if not d:
        return []
    try:
        pts = d["punti"]
        nd = d["non_dominati"]
        chosen = d["scelto"]
    except (KeyError, TypeError) as exc:
        print(f"  [pareto_front.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []
    if len(nd) != len(pts):
        print("  [pareto_front.json ignored: punti and non_dominati have different lengths]",
              file=sys.stderr)
        return []

    def spread(key):
        vals = [p[key] for p in pts]
        return max(vals) - min(vals), min(vals), max(vals)

    M.group("multi-objective Pareto front")
    M.add("resParetoPoints", str(len(pts)), "scalarizzazioni provate")
    M.add("resParetoNondom", str(sum(1 for b in nd if b)), "non-dominated points")
    M.add("resParetoConvex", yesno(d.get("fronte_convesso")))
    M.add("resParetoChosen", "(" + ",\\ ".join(fx(a, 1) for a in chosen) + ")",
          "scalarizzazione scelta")
    sa, _, _ = spread("accuratezza")
    ss, _, _ = spread("sforzo")
    stt, _, _ = spread("tempo")
    M.add("resParetoSpreadAcc", sci(sa), "excursion of the accuracy [m]")
    M.add("resParetoSpreadEffort", sci(ss))
    M.add("resParetoSpreadTime", fx(stt, 2), "excursion of the time to goal [s]")

    # RELATIVE excursion: it is the readable form, and the one that says whether
    # the trade-off really exists. The script also states its own verdict.
    esc_rel = d.get("escursione_relativa") or {}
    for key, name in (("accuratezza", "resParetoRelAcc"),
                      ("sforzo", "resParetoRelEffort"),
                      ("tempo", "resParetoRelTime")):
        if key in esc_rel:
            M.add(name, pc(esc_rel[key], 1), f"escursione relativa: {key} [%]")
    informative = d.get("fronte_informativo")
    M.add("resParetoInformative", yesno(informative))

    # The barycentre of the simplex reproduces the starting tuning: if it is among
    # sampled points, knowing whether it is dominated is the useful result.
    bary = None
    if pts and "alpha" in pts[0]:
        k = len(pts[0]["alpha"])
        target = [1.0 / k] * k
        j = min(range(len(pts)),
                key=lambda i: sum((a - b) ** 2
                                  for a, b in zip(pts[i]["alpha"], target)))
        if sum((a - b) ** 2 for a, b in zip(pts[j]["alpha"], target)) < 0.02:
            bary = (j, bool(nd[j]))
            M.add("resParetoBaryNondom", yesno(bary[1]),
                  "the barycentre (current tuning) is non-dominated")

    if esc_rel:
        rel_txt = (
            f"The relative excursion over the simplex is what says whether a trade-off "
            f"exists at all: accuracy moves by $\\resParetoRelAcc\\%$, effort by "
            f"$\\resParetoRelEffort\\%$ and time to goal by "
            f"$\\resParetoRelTime\\%$. Only the first responds appreciably to the "
            f"weights; the other two are very nearly fixed, because the speed along the "
            f"path is decided by the kinematics and the input bounds rather than by the "
            f"tuning.")
    else:
        rel_txt = (
            f"Over the whole simplex the accuracy moves by $\\resParetoSpreadAcc$~m, "
            f"the effort by $\\resParetoSpreadEffort$ and the time to goal by "
            f"$\\resParetoSpreadTime$~s.")
    if informative is False:
        rel_txt += (
            " The sweep declares the front \\emph{not informative} at this resolution "
            "($\\resParetoInformative$), and that verdict should be carried into the "
            "report as it stands: the table below is a demonstration of the procedure, "
            "not yet evidence of a compromise. A scenario in which the objectives "
            "genuinely conflict is what would turn it into one.")

    if bary is not None and bary[1]:
        bary_txt = (
            f"The useful result is negative. The barycentre of the simplex reproduces the "
            f"tuning already in use, and it comes out \\emph{{non-dominated}} "
            f"($\\resParetoBaryNondom$): no sampled reweighting of the three objectives "
            f"improves one without giving up another. Weight tuning is therefore not the "
            f"bottleneck of this system --- unlike the horizon, which "
            f"\\S\\,\\ref{{res:horizon}} shows to be chosen badly.")
    elif bary is not None:
        bary_txt = (
            f"The barycentre of the simplex --- which reproduces the tuning already in "
            f"use --- is \\emph{{dominated}} on this sweep "
            f"($\\resParetoBaryNondom$), so there is a reweighting that improves at "
            f"least one objective at no cost on the others. That is worth following up "
            f"before the front is used to argue that the current weights are settled.")
    else:
        bary_txt = (
            f"The barycentre of the simplex, which would reproduce the tuning already in "
            f"use, is not among the sampled points at this resolution, so the sweep "
            f"cannot say whether the current weights are dominated. Sampling it "
            f"explicitly is the cheapest way to make this section answer the question the "
            f"report will ask of it.")

    rows = []
    for p, ok in zip(pts, nd):
        a = "(" + ",\\ ".join(fx(x, 1) for x in p["alpha"]) + ")"
        f = (lambda s: r"\textbf{" + s + "}") if ok else (lambda s: s)
        rows.append([f"${a}$", f(fx(p["accuratezza"], 4)), f(fx(p["sforzo"], 4)),
                     f(fx(p["tempo"], 1)), f(fx(p["clearance"], 3)), yesno(ok)])
    return [
        r"\resSubsec{Multi-objective scalarisation and the Pareto front}",
        r"\label{res:pareto}",
        "",
        r"\resNote{Feeds Report \texttt{sec:weights}, which already exhibits one "
        r"two-objective trade-off (effort against accuracy) but selects its operating "
        r"point implicitly. This makes the scalarisation explicit and the front "
        r"measurable.}",
        "",
        f"The cost weights implement a scalarisation of three competing objectives --- "
        f"tracking accuracy, control effort and time to goal --- with fixed coefficients. "
        f"Sweeping the coefficients over the simplex and running each resulting controller "
        f"in closed loop gives $\\resParetoPoints$ points, of which "
        f"$\\resParetoNondom$ are non-dominated (Table~\\ref{{res:tab:pareto}}); the "
        f"front is convex: $\\resParetoConvex$, so the weighted-sum scalarisation can in "
        f"principle reach every point of it.",
        "",
        rel_txt,
        "",
        bary_txt,
        "",
    ] + table("lrrrrc",
              [r"$\kappa$", "accuracy [m]", "effort", "time [s]", "clearance [m]",
               "non-dom."], rows,
              "Closed-loop outcome of each scalarisation of the three objectives. "
              "Bold: non-dominated points.",
              "res:tab:pareto")


# ---------------------------------------------------------------------------
# Sezioni alimentate dagli script satellite
#
# These do not go through results.json: every script writes its own file in
# metrics/out/. They are optional by construction — a missing file skips only
# section, not the document.
# ---------------------------------------------------------------------------
def sec_shooting(extra: dict, M: Macros) -> list[str]:
    rows_in = extra.get("shooting")
    if not rows_in:
        return []
    try:
        per_N: dict[int, dict] = {}
        for r in rows_in:
            per_N.setdefault(int(r["N"]), {})[r["modo"]] = r
        per_N = {k: v for k, v in per_N.items() if "multiple" in v and "single" in v}
        if not per_N:
            return []
    except (KeyError, TypeError) as exc:
        print(f"  [shooting_compare.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []

    Ndep = M.params.get("N")
    Nref = Ndep if Ndep in per_N else max(per_N)
    mu, si = per_N[Nref]["multiple"], per_N[Nref]["single"]

    M.group("single against multiple shooting")
    M.add("resShootN", str(Nref), "horizon of the comparison")
    M.add("resShootVarMulti", str(mu["n_var"]))
    M.add("resShootVarSingle", str(si["n_var"]))
    M.add("resShootConMulti", str(mu["n_con"]))
    M.add("resShootConSingle", str(si["n_con"]))
    M.add("resShootHessDensMulti", fx(100 * mu["hess_density"], 2), "Hessian density [%]")
    M.add("resShootHessDensSingle", fx(100 * si["hess_density"], 2))
    # The minima may differ: the problem is not convex and the two
    # parametrisations follow different paths. It has to be said, not hidden.
    disagreeing = [N for N, v in per_N.items()
                   if abs(v["multiple"]["J"] - v["single"]["J"]) >
                   1e-6 * max(1.0, abs(v["multiple"]["J"]))]
    M.add("resShootSameMinima", yesno(not disagreeing),
          "stesso minimo su tutti gli N")

    rows = []
    for N in sorted(per_N):
        m_, s_ = per_N[N]["multiple"], per_N[N]["single"]
        win = "multiple" if m_["ms"] < s_["ms"] else "single"
        # The Jacobian drops out: it stays sparse in BOTH parametrisations, so it
        # does not discriminate. In its place the ITERATIONS, which separate the
        # work of the solver from the cost of a single iteration — and it is the
        # two that localises the cost in the linear algebra on the full Hessian.
        rows.append([str(N),
                     f'{m_["n_var"]} / {s_["n_var"]}',
                     f'{m_["n_con"]} / {s_["n_con"]}',
                     m(f'{fx(100*m_["hess_density"],2)} / {fx(100*s_["hess_density"],2)}'),
                     f'{m_["iter"]} / {s_["iter"]}',
                     f'{fx(m_["ms"],0)} / {fx(s_["ms"],0)}', win])

    wins_multiple = [N for N in sorted(per_N)
                     if per_N[N]["multiple"]["ms"] < per_N[N]["single"]["ms"]]
    if len(wins_multiple) == len(per_N):
        timing_txt = ("On this run the sparse parametrisation is the faster one at every "
                      "horizon tested")
    elif not wins_multiple:
        timing_txt = ("On this run the condensed parametrisation is the faster one at "
                      "every horizon tested")
    else:
        timing_txt = ("On this run the sparse parametrisation is faster at "
                      + ", ".join(f"$N={n}$" for n in wins_multiple)
                      + " and the condensed one elsewhere")

    L = [
        r"\resSubsec{Condensed against sparse parametrisation of the same program}",
        r"\label{res:shoot}",
        "",
        r"\resNote{Feeds Report \texttt{sec:cost}, which asserts multiple shooting is the "
        r"right choice ``because a condensed single-shooting formulation would produce a "
        r"small but dense problem with a much worse conditioned Hessian''. Half of that "
        r"is now measured and half of it is not: this block reports which half.}",
        "",
        f"The same optimal control problem was built in both parametrisations, with the "
        f"transition map written once and shared, so that what is compared is the "
        f"parametrisation and not two different models. Eliminating the states by "
        f"recursive substitution removes the dynamic equalities altogether: at "
        f"$N=\\resShootN$ the program shrinks from $\\resShootVarMulti$ variables and "
        f"$\\resShootConMulti$ constraints to $\\resShootVarSingle$ and "
        f"$\\resShootConSingle$.",
        "",
        f"The structural half of the claim holds exactly, and the Hessian is where it "
        f"shows: $\\resShootHessDensMulti\\%$ dense in the sparse parametrisation against "
        f"$\\resShootHessDensSingle\\%$ in the condensed one, which is a full matrix. "
        f"Condensing trades a large banded problem for a small dense one, exactly as "
        f"stated.",
        "",
        f"The performance half does not follow from that. {timing_txt} "
        f"(Table~\\ref{{res:tab:shoot}}). Each timing is the fastest of three cold "
        f"solves, so it is a best case rather than a typical one, and none of them uses "
        f"the warm start the controller runs with; the dimensions, the densities and the "
        f"iteration counts beside them are exact. What can be asserted without a stopwatch is the "
        f"conditioning argument: the condensed form integrates the model in open loop "
        f"over the whole horizon, so the error compounds step by step and the problem "
        f"degrades with $N$ and with any instability of the plant. On a kinematic, stable "
        f"model that defect does not surface --- which is precisely why this comparison "
        f"cannot be used to argue the general case, and why the report should claim the "
        f"structure rather than the speed.",
        "",
    ]
    if disagreeing:
        L += [
            f"At $N\\in\\{{{', '.join(str(n) for n in sorted(disagreeing))}\\}}$ the two "
            f"parametrisations converge to \\emph{{different}} minima. This is not an "
            f"implementation error: the program is non-convex, and two parametrisations "
            f"follow different optimization paths, so they can land in different basins. "
            f"It does mean the corresponding timings compare two solves that did not "
            f"solve the same thing.",
            "",
        ]
    return L + table("rlllll",
                     ["$N$", "vars M / S", "cons M / S",
                      "hess dens.\\ [\\%] M / S", "iter M / S", "solve [ms] M / S"],
                     [r[:6] for r in rows],
                     "Multiple (M) against single (S) shooting on the same instance. "
                     "Dimensions, densities and iteration counts are exact; each timing is "
                     "the fastest of three cold solves, with no warm start.",
                     "res:tab:shoot")


def sec_solver_compare(extra: dict, M: Macros) -> list[str]:
    rows_in = extra.get("solver")
    if not rows_in:
        return []
    try:
        rows_in = sorted(rows_in, key=lambda r: r["n_ineq"])
        for r in rows_in:
            r["ipopt"], r["sqp"], r["n_ineq"]
    except (KeyError, TypeError) as exc:
        print(f"  [solver_compare.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []

    M.group("interior point against active set")
    n_lo, n_hi = rows_in[0]["n_ineq"], rows_in[-1]["n_ineq"]
    M.add("resSolverIneqLow", str(n_lo), "disuguaglianze, regime piccolo")
    M.add("resSolverIneqHigh", str(n_hi), "disuguaglianze, regime grande")

    # A ratio of times only makes sense if BOTH solvers return a solution: where
    # SQP fails, the "time" is time spent before giving up and the ratio is not a
    # speed-up. The failed rows go into the failure count, not into the means.

    def _ratios(n_ineq):
        return [r["sqp"]["ms"] / r["ipopt"]["ms"] for r in rows_in
                if r["n_ineq"] == n_ineq and r["ipopt"].get("ok") and r["sqp"].get("ok")]

    r_lo, r_hi = _ratios(n_lo), _ratios(n_hi)
    if r_lo:
        M.add("resSolverRatioMin", fx(min(r_lo), 1),
              "smallest interior-point advantage, small regime")
        M.add("resSolverRatioLow", fx(max(r_lo), 1),
              "interior-point advantage, small regime")
    if r_hi:
        M.add("resSolverRatioHigh", fx(max(r_hi), 1),
              "interior-point advantage, large regime")
    n_fail = sum(1 for r in rows_in if not r["sqp"].get("ok"))
    M.add("resSolverSqpFail", str(n_fail), "runs where SQP does not solve the QP")
    M.add("resSolverRuns", str(len(rows_in)), "runs of the comparison")
    all_same = all(r.get("stesso_minimo") for r in rows_in
                   if r["ipopt"].get("ok") and r["sqp"].get("ok"))
    M.add("resSolverSameMinima", yesno(all_same))

    def _ms(v, vince):
        t = fx(v, 0)
        return r"\textbf{" + t + "}" if vince else t
    def _sqp_cell(r):
        """A failed run has neither iterations nor solve time: it has to be marked
        as a failure, not printed as if it were a comparison."""
        if not r["sqp"].get("ok"):
            ms = r["sqp"]["ms"]
            t = f"{ms/1000:.1f} s" if ms >= 1000 else f"{fx(ms, 0)} ms"
            return r"\emph{fail} (" + t + ")"
        return f'{_ms(r["sqp"]["ms"], r["sqp"]["ms"] < r["ipopt"]["ms"])} / {r["sqp"]["iter"]}'

    rows = [[solver_regime(r["regime"]), str(r.get("caso", "")), str(r["n_ineq"]),
             f'{_ms(r["ipopt"]["ms"], r["ipopt"]["ms"] <= r["sqp"]["ms"])} / {r["ipopt"]["iter"]}',
             _sqp_cell(r),
             m(fx(r["sqp"]["ms"] / r["ipopt"]["ms"], 1) + r"\times")
             if r["sqp"].get("ok") else "---"] for r in rows_in]

    L = [
        r"\resSubsec{Interior point against active set}",
        r"\label{res:solvercmp}",
        "",
        r"\resNote{Feeds Report \texttt{sec:solver}, which argues for an interior-point "
        r"method on the grounds that the constraint set is dominated by equalities. That "
        r"argument is now testable rather than asserted, because the obstacle formulation "
        r"is switchable and the same system can be put in both regimes.}",
        "",
        f"The rule of thumb is that active-set methods win when the inequalities are few "
        f"and interior-point methods win when they are many. This stack can be moved "
        f"between the two regimes without changing anything else: with the obstacles in "
        f"the objective the program carries $\\resSolverIneqLow$ inequalities, and with "
        f"the obstacles as genuine constraints it carries $\\resSolverIneqHigh$.",
        "",
        f"The rule does not reproduce (Table~\\ref{{res:tab:solvercmp}}): the active-set "
        f"solver does not win even in the small regime, where the interior-point method "
        f"is already $\\resSolverRatioLow\\times$ faster. The \\emph{{direction}} is "
        f"confirmed --- the margin widens to $\\resSolverRatioHigh\\times$ when the "
        f"inequalities multiply --- so the break-even point, if there is one, lies below "
        f"the smallest regime this problem can be put in.",
        "",
        r"Two cautions, without which the numbers would be misleading. First, the real "
        r"advantage of an active set in MPC is warm starting \emph{between consecutive "
        r"solves}: the active set changes by a few rows per cycle and the factorisation "
        r"is reused. Both solvers are started cold here, on purpose, so as to favour "
        r"neither --- which removes from the active-set method exactly what makes it "
        r"competitive in a receding-horizon loop. Second, the rule of thumb is stated for "
        r"convex programs, and this one is not.",
        "",
        r"A by-product worth reporting: with the exact Hessian of the Lagrangian the QP "
        r"subproblem is repeatedly flagged indefinite. That is expected and instructive "
        r"--- a non-convex program can produce a non-convex QP, which has no unique "
        r"solution --- and it is the reason a Gauss-Newton Hessian, positive "
        r"semi-definite by construction, is the standard recommendation for SQP.",
        "",
    ]
    if not all_same:
        L += [r"\resNote{At least one pair of solves converged to different objective "
              r"values. Comparing the time of two solves that ended in different minima "
              r"means nothing; that row is not evidence for either method.}", ""]
    return L + table("llrrrr",
                     ["regime", "instance", "ineq.", "IPOPT ms / iter",
                      "SQP+qpOASES ms / iter", "speed-up"], rows,
                     "The same instance solved by a primal--dual interior-point method "
                     "and by an SQP with an active-set QP solver, in both obstacle "
                     "regimes. Both are started cold. Bold: the faster of the two.",
                     "res:tab:solvercmp")


def sec_control_horizon(extra: dict, M: Macros) -> list[str]:
    rows_in = extra.get("control")
    if not rows_in:
        return []
    try:
        for r in rows_in:
            r["N"], r["N_c"], r["p95"], r["t_goal"], r["clearance"]
    except (KeyError, TypeError) as exc:
        print(f"  [control_horizon.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []

    # Pairs (same scenario, same N) between the smallest N_c and N_c = N: that is
    # reads what the degrees of freedom cost at a fixed horizon.
    by = {}
    for r in rows_in:
        by.setdefault((str(r.get("scenario", "")), int(r["N"])), []).append(r)
    gains = []
    for (scen, N), group in by.items():
        full = next((g for g in group if int(g["N_c"]) == N), None)
        red = min((g for g in group if int(g["N_c"]) < N),
                  key=lambda g: int(g["N_c"]), default=None)
        if not full or not red:
            continue
        same = (abs(full["t_goal"] - red["t_goal"]) < 1e-6
                and abs(full["clearance"] - red["clearance"]) < 1e-3)
        gains.append({"scen": scen, "N": N, "N_c": int(red["N_c"]),
                      "ratio": full["p95"] / max(red["p95"], 1e-9),
                      "same": same, "p95_full": full["p95"], "p95_red": red["p95"]})
    free = [g for g in gains if g["same"] and g["ratio"] > 1.0]

    # The diagnostic question ("does the degradation come from the prediction or
    # from the degrees of freedom?") only makes sense if the sweep really contains
    # a horizon that degrades. If it does not, say so instead of reporting the
    # conclusion.
    per_scen: dict[str, list] = {}
    for r in rows_in:
        per_scen.setdefault(str(r.get("scenario", "")), []).append(r)
    degrading = []
    for scen, group in per_scen.items():
        Ns = sorted({int(g["N"]) for g in group})
        if len(Ns) < 2:
            continue
        base = min(g["t_goal"] for g in group if int(g["N"]) == Ns[0])
        for N in Ns[1:]:
            worst = max(g["t_goal"] for g in group if int(g["N"]) == N)
            if worst > base + 1e-6:
                degrading.append((scen, N))
    tight = [r for r in rows_in if float(r["clearance"]) < 0.01]

    M.group("control horizon N_c")
    M.add("resNcCases", str(len(rows_in)), "configurazioni provate")
    if gains:
        best = max(gains, key=lambda g: g["ratio"])
        M.add("resNcBestRatio", fx(best["ratio"], 1), "best p95 saving")
        M.add("resNcBestN", str(best["N"]))
    M.add("resNcFreeCases", str(len(free)),
          "cases where cutting the degrees of freedom is free")

    rows = []
    for r in sorted(rows_in, key=lambda r: (str(r.get("scenario", "")), r["N"], r["N_c"])):
        rows.append([esc(r.get("scenario", "")), str(r["N"]), str(r["N_c"]),
                     str(r["n_var"]), yesno(r["goal"]), fx(r["t_goal"], 1),
                     fx(r["clearance"], 3), fx(r["p95"], 1)])

    L = [
        r"\resSubsec{Control horizon: separating preview from degrees of freedom}",
        r"\label{res:nc}",
        "",
        r"\resNote{New material, and it answers a question \S\,\ref{res:horizon} cannot: "
        r"there $N$ governs both how far the controller looks and how many free inputs it "
        r"has, so a degradation with $N$ has two candidate causes. Belongs next to "
        r"\texttt{sec:horizon}.}",
        "",
        f"Freeing only the first $N_c$ inputs and holding the rest at the last free value "
        f"decouples the two roles of the horizon: the prediction still runs to $N$, but "
        f"the program carries fewer variables. Over $\\resNcCases$ configurations "
        f"(Table~\\ref{{res:tab:nc}}) the two effects separate cleanly.",
        "",
    ]
    if free:
        L += [
            f"Where the prediction horizon is already the right length, cutting the "
            f"degrees of freedom is free: in $\\resNcFreeCases$ of the paired cases the "
            f"time to goal and the minimum clearance are unchanged while the 95th "
            f"percentile of the solve time falls, by up to "
            f"$\\resNcBestRatio\\times$ at $N=\\resNcBestN$. This is the input "
            f"parametrisation argument in its cheapest form: the degrees of freedom past "
            f"the first few are not what the closed loop was using.",
            "",
        ]
    if degrading:
        L += [
            r"The complementary reading is the one that matters for "
            r"\S\,\ref{res:horizon}: where a long horizon degrades the closed loop, "
            r"reducing $N_c$ at the same $N$ does \emph{not} recover the short-horizon "
            r"behaviour. The degradation therefore comes from the prediction --- the "
            r"reference extends over a path that the discrete planner will replan "
            r"anyway, so the controller commits to a target due to change --- and not "
            r"from an excess of decision variables. A long prediction horizon cannot be "
            r"bought cheaply: if terminal ingredients required one, it would cost "
            r"closed-loop performance and not only computation.",
            "",
        ]
    else:
        L += [
            r"The complementary question --- whether the degradation a long horizon "
            r"causes in \S\,\ref{res:horizon} comes from the preview or from the extra "
            r"degrees of freedom --- is \emph{not} answered by this sweep: none of the "
            r"horizons tested here degrades the closed loop, so there is no degradation "
            r"to attribute. Answering it requires re-running this comparison over the "
            r"horizons that do degrade, which is the cheapest missing measurement in "
            r"this document.",
            "",
        ]
    L += [
        r"\resNote{\textbf{On move blocking.} Holding the input constant over blocks of "
        r"increasing length is the standard way to recover computation without shortening "
        r"the horizon, and it is deliberately not used here. Wherever \S\,\ref{res:horizon} "
        r"finds the useful horizon to be short, compressing a handful of variables into "
        r"blocks changes nothing measurable; the control horizon above is the degenerate "
        r"case of move blocking --- one free block followed by one long one --- and "
        r"already delivers the saving; and computation is not the binding resource, since "
        r"the deployed configuration sits well inside its cycle budget while the "
        r"prediction error of \S\,\ref{res:pred} does not. It is a considered choice, "
        r"not an omission, and it would become the right technique if a rigorous terminal "
        r"ingredient forced a long horizon.}",
        "",
        r"One implementation detail carries theory. Imposing the input bounds on all $N$ "
        r"steps when the inputs past $N_c$ are the same repeated expression generates "
        r"duplicate rows with identical gradients; if active, they violate LICQ and make "
        r"the multipliers non-unique --- breaking exactly the analysis of "
        r"\S\,\ref{res:kkt}. The bounds must be imposed on the free columns only.",
        "",
    ]
    if tight:
        scen_t = sorted({str(r.get("scenario", "")) for r in tight})
        L += [
            f"One reading of the table is not about $N_c$ at all and should not be "
            f"passed over: in {', '.join(tt(x) for x in scen_t)} the minimum clearance "
            f"is zero to display precision in every configuration, so the robot grazes "
            f"an obstacle regardless of the control horizon. Whatever that scenario is "
            f"testing, it is not being solved safely, and the comparison above is made "
            f"between configurations that all touch.",
            "",
        ]
    return L + table("lrrrcrrr",
                     ["scenario", "$N$", "$N_c$", "vars", "goal", "TTG [s]",
                      "min clear.\\ [m]", "p95 [ms]"], rows,
                     "Closed-loop outcome against the control horizon at fixed prediction "
                     "horizon.",
                     "res:tab:nc")


def sec_robust(extra: dict, M: Macros) -> list[str]:
    d = extra.get("robust")
    if not d:
        return []
    try:
        beta = [float(b) for b in d["beta"]]
        q = float(d["quantile"])
        rows_in = d["righe"]
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  [robust_constraints.json ignorato: schema inatteso ({exc})]", file=sys.stderr)
        return []

    dt = M.params.get("dt", 0.0)
    monotone = all(b <= a + 1e-12 for a, b in zip(beta[1:], beta[2:])) or \
        all(beta[i] <= beta[i + 1] + 1e-12 for i in range(len(beta) - 1))

    M.group("robust constraints (constraint tightening)")
    M.add("resBetaQuantile", fx(100 * q, 0), "quantile used for the tube [%]")
    M.add("resBetaZero", fx(beta[0], 4), "margine a k=0 [m]")
    M.add("resBetaEnd", fx(beta[-1], 4), "margine a fine orizzonte [m]")
    M.add("resBetaMonotone", yesno(monotone))
    eff = [r for r in rows_in if "efficace" in str(r.get("esito", ""))]
    M.add("resRobustCases", str(len(rows_in)), "casi provati")
    M.add("resRobustEffective", str(len(eff)), "cases where the tube bites")
    if eff:
        M.add("resRobustBestDelta", fx(max(r["delta"] for r in eff), 3),
              "largest clearance gain [m]")

    # Every observed outcome reads differently, and only those that
    # the data really contain.
    kinds = {outcome(r.get("esito", "")) for r in rows_in}
    frasi = []
    if "constraint inactive" in kinds:
        frasi.append("where $d_{\\mathrm{safe}}+\\gamma$ stays below the clearance the "
                     "trajectory already keeps, the constraint is inactive and correctly "
                     "does nothing")
    if any("effective" in k for k in kinds):
        frasi.append("where it bites, the predicted clearance increases with the slack "
                     "still exactly zero, so the margin is \\emph{respected} rather than "
                     "violated and paid for")
    if "infeasible" in kinds:
        frasi.append("and where it asks for more than the input set can deliver, the "
                     "$\\ell^1$ penalty of \\S\\,\\ref{res:penalty} yields instead of "
                     "rendering the program infeasible --- which is precisely why that "
                     "relaxation was chosen")
    if len(frasi) > 1:
        outcomes_txt = ("the outcomes separate cleanly and each is informative: "
                        + "; ".join(frasi) + ".")
    elif frasi:
        outcomes_txt = frasi[0].capitalize() + ("."
            " The other two regimes --- a tube that bites, and one that asks for more"
            " than the input set can deliver --- do not occur in this sweep, so the"
            " remaining behaviour of the tightening is still untested.")
    else:
        outcomes_txt = ("no outcome could be classified, which means the sweep needs "
                        "re-running before this block says anything.")

    brows = [[str(k), fx(k * dt, 1), fx(b, 4)] for k, b in enumerate(beta)]
    rrows = [[esc(r.get("scenario", "")), fx(r["d_safe"], 2),
              fx(r["senza"]["clearance"], 4), fx(r["con"]["clearance"], 4),
              fx(r["delta"], 4),
              f'{smart(r["senza"]["slack"])} / {smart(r["con"]["slack"])}',
              outcome(r.get("esito", ""))] for r in rows_in]

    L = [
        r"\resSubsec{Constraint tightening from the measured prediction error}",
        r"\label{res:robust}",
        "",
        r"\resNote{New material. Feeds Report \texttt{sec:barrier} and "
        r"\texttt{sec:mismatch}: the clearance constraint is imposed on the "
        r"\emph{predicted} trajectory, and \S\,\ref{res:pred} measures how far that "
        r"diverges from the executed one. This closes the gap instead of noting it.}",
        "",
        f"The obstacle constraint is tightened by a margin $\\gamma(k)$ that grows along "
        f"the horizon, $\\lVert p_k-o_j\\rVert \\ge d_{{\\mathrm{{safe}}}}+\\gamma(k)-s_{{jk}}$. "
        f"The margin is not postulated: it is read off the "
        f"$\\resBetaQuantile$th percentile of the prediction error recorded in the run "
        f"of \\S\\,\\ref{{res:pred}}, so the tube is derived from data rather than from an "
        f"assumption about the disturbance.",
        "",
        f"Three properties hold by construction and are worth checking rather than "
        f"assuming (Table~\\ref{{res:tab:beta}}). $\\gamma(0)=\\resBetaZero$ exactly: at "
        f"the first node the state is fixed by an equality constraint, so there is "
        f"nothing to hedge against and the constraint is not tightened where it would "
        f"only remove feasible motion. The margin is monotone "
        f"($\\resBetaMonotone$), as uncertainty is. And it reaches "
        f"$\\resBetaEnd$~m at the end of the horizon, which is the same order as the "
        f"safety radius itself --- the tube is not a decoration.",
        "",
        f"Measured on the predicted trajectory ($\\resRobustCases$ cases, "
        f"Table~\\ref{{res:tab:robust}}), {outcomes_txt}",
        "",
        r"\resNote{\textbf{Limit of this measurement, to be stated in the report.} The "
        r"effect is not observable in the closed-loop harness: the executed clearance "
        r"comes out identical with and without tightening, because the loop is closed by "
        r"tracking a look-ahead setpoint on the predicted trajectory with a proportional "
        r"controller, which washes out fine differences between MPC solutions. The "
        r"tightening guarantees the margin \emph{in the plan}, and that is where it has "
        r"been verified.}",
        "",
    ]
    L += table("rrr", ["$k$", "$k\\,\\Delta t$ [s]", r"$\gamma(k)$ [m]"], brows,
               f"Constraint back-off derived from the ${fx(100*q,0)}$th percentile of the "
               f"measured prediction error. The offset at $k=0$ is subtracted: it is "
               f"time misalignment, not model uncertainty, and including it would inflate "
               f"the tube by a constant.",
               "res:tab:beta")
    return L + table("lrrrrll",
                     ["scenario", "$d_{\\mathrm{safe}}$", "clear.\\ without",
                      "clear.\\ with", "$\\Delta$", "slack w/o / with", "outcome"],
                     rrows,
                     "Effect of the tightening on the predicted trajectory.",
                     "res:tab:robust")


# ---------------------------------------------------------------------------
# Body assembly
#
# The order is that of the report, not that of the code: model ->
# discretisation -> NLP -> derivatives -> optimality -> regularity ->
# reformulations -> closed-loop campaigns. The correspondence table at the top
# of the document is built from this same list, so it cannot diverge from the
# sections actually present.
# ---------------------------------------------------------------------------
SPECS = [
    (sec_discretisation, "res:disc", "Model discretisation, truncation order",
     "sec:discretization"),
    (sec_prediction, "res:pred", "Open-loop prediction error",
     "sec:model, sec:mismatch"),
    (sec_nlp, "res:nlp", "NLP size and sparsity", "sec:dims (tab:nlp)"),
    (sec_shooting, "res:shoot", "Condensed against sparse parametrisation", "sec:cost"),
    (sec_derivatives, "res:ad", "AD against finite differences",
     "sec:solver, sec:impl"),
    (sec_hessian, "res:hess", "Exact Hessian against L-BFGS", "sec:solver"),
    (sec_solver_compare, "res:solvercmp", "Interior point against active set",
     "sec:solver"),
    (sec_kkt, "res:kkt", "KKT, LICQ, second-order conditions",
     "new, next to sec:constraints"),
    (sec_penalty, "res:penalty", "Exact $\\ell^1$ penalty", "new, next to sec:barrier"),
    (sec_terminal, "res:terminal", "Terminal equilibrium constraint", "sec:terminal"),
    (sec_robust, "res:robust", "Constraint tightening from measured error",
     "new, next to sec:barrier / sec:mismatch"),
    (sec_bifurcation, "res:bif", "Solution regularity, bifurcation", "sec:barriersweep"),
    (sec_pathfollowing, "res:pf", "Path-parametrised reference", "sec:refwarm"),
    (sec_horizon, "res:horizon", "Horizon and sampling time",
     "sec:horizon, sec:dtsweep"),
    (sec_control_horizon, "res:nc", "Control horizon, and why not move blocking",
     "new, next to sec:horizon"),
    (sec_pareto, "res:pareto", "Multi-objective scalarisation", "sec:weights"),
]

_EXTRA_SECTIONS = {sec_horizon, sec_pareto, sec_solver_compare, sec_shooting,
                   sec_control_horizon, sec_robust}

BODY_HEADER = r"""% ============================================================================
% metrics_body.tex — GENERATO AUTOMATICAMENTE, NON MODIFICARE A MANO
%
%   regenerate with: python3 metrics/make_results.py
%             or with: python3 metrics/results_tex.py
%
% This file has no preamble: it is meant to be \input{} inside Report.tex.
% It requires the packages the report already loads (booktabs, amsmath, amssymb).
%
% WHEN INTEGRATING INTO THE REPORT, three lines are enough:
%   \renewcommand{\resSec}[1]{\subsection{#1}}     % demote the headings
%   \renewcommand{\resSubsec}[1]{\subsubsection{#1}}
%   \renewcommand{\resNote}[1]{}                   % drop the service notes
% ============================================================================

% --- scaffolding: redefinable from outside ---------------------------------
\providecommand{\resSec}[1]{\section{#1}}
\providecommand{\resSubsec}[1]{\subsection{#1}}
\providecommand{\resNote}[1]{{\small\itshape #1\par\medskip}}

% --- report symbols: \providecommand, so the report wins if it has them ---
\providecommand{\R}{\mathbb{R}}
\providecommand{\Wobs}{W_{\mathrm{obs}}}
\providecommand{\robs}{r_{\mathrm{obs}}}
\providecommand{\aobs}{\alpha_{\mathrm{obs}}}
\providecommand{\Jobs}{J_{\mathrm{obs}}}
\providecommand{\astar}{$A^\star$}
"""

STANDALONE = r"""% ============================================================================
% metrics_standalone.tex — AUTOMATICALLY GENERATED
%
% Minimal wrapper to compile the metrics ON THEIR OWN, without touching Report.tex:
%
%     pdflatex metrics_standalone.tex
%
% When the sections are moved into the report, this file is thrown away and
% only metrics_body.tex + metrics_macros.tex are kept.
% ============================================================================
\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{caption}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black, urlcolor=black, citecolor=black}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}

\title{Measured quantities of the optimization problem\\[2pt]
       \large A\textsuperscript{$\star$} + nonlinear MPC navigation stack, Unitree G1}
\author{Auto-generated from \texttt{metrics/out/results.json}}
\date{\resDate}

\input{metrics_macros}

\begin{document}
\maketitle
\thispagestyle{empty}

\noindent\textbf{What this document is.} A staging file. Every number here is
produced by \texttt{metrics/make\_results.py} from the same modules the deployed
planner imports, and is meant to be moved into the report section named beside
it. Nothing in it is written by hand, and it is not itself a chapter.

\tableofcontents

\input{metrics_body}

\end{document}
"""


def build_body(res: dict, extra: dict, M: Macros) -> str:
    L = sec_provenance(res, M)
    rendered: list[tuple[str, str, str]] = []
    tail: list[str] = []
    for fn, label, title, target in SPECS:
        block = fn(extra, M) if fn in _EXTRA_SECTIONS else fn(res, M)
        if not block:
            continue
        rendered.append((label, title, target))
        tail += block

    # Correspondence table: built from the sections actually
    # present, so that a --only classe1 does not promise blocks that are missing.
    L += [
        r"\resSubsec{Where each block belongs in the report}",
        "",
        r"\resNote{This table is the point of the file: it is the integration plan. "
        r"Delete it once the blocks have been moved.}",
        "",
    ]
    L += table("llp{0.40\\textwidth}",
               ["block", "content", "target section in the report"],
               [[f"\\S\\,\\ref{{{lab}}}", esc(t) if "$" not in t else t, tt(tgt)]
                for lab, t, tgt in rendered],
               "Destination of each block of measurements. The report sections are "
               "named by their label, not by their number, because the numbering is "
               "still going to change.",
               "res:tab:map", small=True)
    L = [BODY_HEADER] + L + tail
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Verifica sintattica minima
#
# It does not replace pdflatex: it catches the mistakes a generator actually
# makes (unbalanced braces, unclosed environments, columns that do not add up).
# ---------------------------------------------------------------------------
def check(text: str, name: str) -> list[str]:
    problems = []
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                problems.append(f"{name}: one closing brace too many at offset {i}")
                depth = 0
    if depth:
        problems.append(f"{name}: {depth} braces opened and not closed")

    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        elif not stack:
            problems.append(f"{name}: \\end{{{env}}} senza \\begin")
        elif stack[-1] != env:
            problems.append(f"{name}: \\end{{{env}}} chiude \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    for env in stack:
        problems.append(f"{name}: \\begin{{{env}}} mai chiuso")

    # math mode: sci()/smart() produce \times and ^{...}, which outside $...$
    # stop LaTeX. The definition lines are excluded by convention: the macros
    # already expand inside $...$ in the text that uses them.
    _MATH = (r"\times", r"\ell", r"\mathrm", r"\lambda", r"\rho", r"\Delta",
             r"\mu", r"\alpha", r"\omega", r"\sqrt", r"\varepsilon", r"\mathcal")
    for ln, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("%") or re.match(r"\s*\\(providecommand|resdef|newcommand|renewcommand)", line):
            continue
        dollars = [mm.start() for mm in re.finditer(r"(?<!\\)\$", line)]
        if len(dollars) % 2:
            problems.append(f"{name}:{ln}: odd number of $")
            continue
        def outside(pos):
            return sum(1 for d in dollars if d < pos) % 2 == 0
        for tok in _MATH:
            for mm in re.finditer(re.escape(tok), line):
                if outside(mm.start()):
                    problems.append(
                        f"{name}:{ln}: {tok} fuori da math mode: {line.strip()[:70]}")
                    break
        for mm in re.finditer(r"(?<!\\)\^", line):
            if outside(mm.start()):
                problems.append(f"{name}:{ln}: ^ fuori da math mode: {line.strip()[:70]}")
                break

    # column count consistent between the specification and the rows
    for m in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)"
                         r"\\end\{tabular\}", text, re.S):
        spec, body = m.group(1), m.group(2)
        # strip the content of the braces (p{...}, >{...}, @{...}): inside there
        # are letters that are not columns, \textwidth being the classic example
        bare, depth = [], 0
        for ch in spec:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
            elif depth == 0:
                bare.append(ch)
        ncol = len(re.findall(r"[lcrpmbX]", "".join(bare)))
        # A table row is delimited by \\, not by the newline of the source: with
        # p{} columns the text wraps and every physical line would look like a
        # table row with a single cell. So it is split on the separator.
        senza_commenti = re.sub(r"(?<!\\)%.*", "", body)
        for chunk in re.split(r"\\\\(?:\[[^\]]*\])?", senza_commenti):
            chunk = chunk.strip()
            if not chunk or chunk.startswith("\\") and "&" not in chunk:
                continue                    # \toprule, \midrule, \bottomrule
            got = len(re.split(r"(?<!\\)&", chunk))
            if got != ncol:
                problems.append(
                    f"{name}: row with {got} cells in a tabular of {ncol} "
                    f"colonne: {' '.join(chunk.split())[:70]}")
    return problems


def check_cross(body: str, macros: str) -> list[str]:
    """
    Checks that need the two files together.

    The typical way this generator can break silently is a typo in the name of a
    macro inside the prose: LaTeX stops with
    "Undefined control sequence" and the file looks fine to the eye. Here it is
    caught before writing.
    """
    problems = []
    scaffold = {"resSec", "resSubsec", "resNote", "resdef", "restab", "restabdir"}
    definite = set(re.findall(r"\\resdef\{(res[A-Za-z]+)\}", macros))
    usate = set(re.findall(r"\\(res[A-Za-z]+)", body)) - scaffold
    for name in sorted(usate - definite):
        problems.append(f"macro used in the body but not defined: \\{name}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for ref in sorted(set(re.findall(r"\\ref\{([^}]+)\}", body))):
        if ref not in labels:
            problems.append(f"\\ref{{{ref}}} senza \\label corrispondente")
    return problems


# ---------------------------------------------------------------------------
# Scrittura
# ---------------------------------------------------------------------------
def load_extra(results_path: str) -> dict:
    """
    Collects the JSON files produced by the satellite scripts sitting in the same
    folder. They are optional by choice: horizon_sweep.py and pareto_front.py
    are still work in progress, so their absence is not an error and a change of
    schema only drops the section concerned, not the file.
    """
    out = {}
    d = os.path.dirname(os.path.abspath(results_path))
    for key, fname in (("horizon", "horizon_sweep.json"),
                       ("pareto", "pareto_front.json"),
                       ("solver", "solver_compare.json"),
                       ("shooting", "shooting_compare.json"),
                       ("control", "control_horizon.json"),
                       ("robust", "robust_constraints.json")):
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                out[key] = json.load(fh)
        except Exception as exc:
            print(f"  [{fname} illeggibile: {exc}]", file=sys.stderr)
            continue
        # The seven satellites are NOT recomputed by make_results: it reads their
        # cache. If the profile changed after the cache was written, the numbers
        # in the report belong to a different configuration, and nothing flags it.
        # It has happened with the input envelope. Comparing the dates is not a
        # proof, but it turns a silent failure into a warning.
        # silent failure into a question.
        prof = os.path.join(os.path.dirname(os.path.dirname(d)),
                            "src", "a_star_mpc_planner", "config",
                            "planner_params_g1.yaml")
        if os.path.exists(prof) and os.path.getmtime(p) < os.path.getmtime(prof):
            print(f"  [WARNING: {fname} is OLDER than the profile "
                  f"({_eta(p)} against {_eta(prof)}). It is a cache: make_results "
                  f"does not recompute it. Re-run the satellite script before "
                  f"quoting its numbers.]", file=sys.stderr)
    return out


def _eta(path: str) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(
        os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def write_all(res: dict, out_dir: str, extra: dict | None = None) -> list[str]:
    """Generate the three files and return their paths. Raises if the .tex is broken."""
    extra = extra or {}
    M = Macros()
    TABLES.clear()                      # rigenerazioni ripetute nello stesso processo
    body = build_body(res, extra, M)
    macros = M.render(res["meta"])
    tabelle = {k: "\n".join(v) + "\n" for k, v in TABLES.items()}

    problems = check(body, "metrics_body.tex") + check(macros, "metrics_macros.tex")
    for k, t in tabelle.items():
        problems += check(t, f"tab/{k}.tex")
    # the cross-check must see the captions too: they use \resBag
    problems += check_cross(body + "\n".join(tabelle.values()), macros)
    if problems:
        raise RuntimeError("generated LaTeX is not valid:\n  " + "\n  ".join(problems))

    os.makedirs(out_dir, exist_ok=True)
    tab_dir = os.path.join(out_dir, "tab")
    os.makedirs(tab_dir, exist_ok=True)
    # Tables that disappear from one regeneration to the next have to be removed,
    # or they stay up there being included by a \restab nobody generates any more.
    for stale in os.listdir(tab_dir):
        if stale.endswith(".tex") and stale[:-4] not in tabelle:
            os.remove(os.path.join(tab_dir, stale))

    paths = []
    for name, text in (("metrics_macros.tex", macros),
                       ("metrics_body.tex", body),
                       ("metrics_standalone.tex", STANDALONE)):
        p = os.path.join(out_dir, name)
        with open(p, "w") as fh:
            fh.write(text)
        paths.append(p)
    for k, t in sorted(tabelle.items()):
        p = os.path.join(tab_dir, f"{k}.tex")
        with open(p, "w") as fh:
            fh.write(t)
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(_HERE, "out", "results.json"),
                    help="JSON prodotto da make_results.py")
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "tex"),
                    help="destination folder of the .tex files")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignora horizon_sweep.json e pareto_front.json")
    ap.add_argument("--check", action="store_true",
                    help="genera in memoria e verifica soltanto, senza scrivere")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} is missing: run  python3 metrics/make_results.py  first",
              file=sys.stderr)
        return 1
    with open(args.results) as fh:
        res = json.load(fh)
    extra = {} if args.no_extra else load_extra(args.results)

    if args.check:
        # Same assembly as when writing, tables included: the \label now live in
        # the tab/ files, so a check on the body alone
        # would report every \ref to a table as an orphan.
        M = Macros()
        TABLES.clear()
        body = build_body(res, extra, M)
        macros = M.render(res["meta"])
        tabelle = {k: "\n".join(v) + "\n" for k, v in TABLES.items()}
        problems = check(body, "body") + check(macros, "macros")
        for k, t in tabelle.items():
            problems += check(t, f"tab/{k}.tex")
        problems += check_cross(body + "\n".join(tabelle.values()), macros)
        for pr in problems:
            print("  " + pr, file=sys.stderr)
        print("verifica fallita" if problems else "verifica superata")
        return 1 if problems else 0

    paths = write_all(res, args.out, extra)
    print("generated:")
    for p in paths:
        print(f"  {os.path.relpath(p, _ROOT)}")
    if res["meta"].get("git_albero_sporco"):
        print("\nWARNING: results.json was produced with a dirty tree;")
        print("the .tex states it at the top, but the right thing is to regenerate it clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
