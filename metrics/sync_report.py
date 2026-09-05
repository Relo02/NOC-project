#!/usr/bin/env python3
"""
sync_report — porta le metriche generate dentro l'albero di compilazione del report.

`make_results.py` scrive macro e tabelle in metrics/out/tex/ e le figure in metrics/out/.
Il report LaTeX le legge da <build>/Metrics/{metrics_macros.tex, tab/, fig/}.
Questo modulo e' il ponte fra i due, ed e' l'unico posto in cui vive la mappa
dai nomi locali delle figure (che dipendono dalla bag e dallo scenario) ai nomi
stabili che il .tex cita. La cartella di destinazione si passa con --dest: il default vale se l'albero
LaTeX sta dentro il repo, in report/.

    python3 metrics/sync_report.py                    # dopo un make_results
    python3 metrics/sync_report.py --dest <cartella>  # verso un altro albero
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

# (glob nella cartella delle figure, nome stabile citato dal .tex)
FIGURES = (
    ("errore_predizione_*.pdf",         "prediction_error.pdf"),
    ("biforcazione_centred_pillar.pdf", "bifurcation.pdf"),
    ("horizon_sweep.pdf",               "horizon_sweep.pdf"),
    ("pareto_front.pdf",                "pareto_front.pdf"),
    ("pannello1_bag_*.pdf",             "cost_landscape.pdf"),
    ("pannello2_*_merit.pdf",           "decision_plane.pdf"),
    ("fig_grid_profile_g1.pdf",         "fig_grid_profile_g1.pdf"),
    ("fig_local_target.pdf",            "fig_local_target.pdf"),
    ("fig_barrier_shape.pdf",           "fig_barrier_shape.pdf"),
)

DEST_DEFAULT = os.path.join(_REPO, "report", "Metrics")


def sync(tex_dir: str, fig_dir: str, dest: str = DEST_DEFAULT,
         verbose: bool = True) -> list[str]:
    """
    Copia macro, tabelle e figure in `dest`. Restituisce i percorsi scritti.

    Non crea l'albero di compilazione da zero: se la cartella che lo contiene
    non esiste si limita a segnalarlo. Sincronizzare dentro una cartella che
    nessuno compila e' peggio che non sincronizzare, perche' sembra fatto.
    """
    parent = os.path.dirname(os.path.abspath(dest))
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"albero del report assente: {parent} — niente da aggiornare")

    scritti: list[str] = []

    src_macros = os.path.join(tex_dir, "metrics_macros.tex")
    if not os.path.isfile(src_macros):
        raise FileNotFoundError(f"macro non generate: {src_macros}")
    os.makedirs(dest, exist_ok=True)
    dst = os.path.join(dest, "metrics_macros.tex")
    shutil.copyfile(src_macros, dst)
    scritti.append(dst)

    tabs = sorted(glob.glob(os.path.join(tex_dir, "tab", "*.tex")))
    if not tabs:
        raise FileNotFoundError(f"nessuna tabella in {tex_dir}/tab")
    os.makedirs(os.path.join(dest, "tab"), exist_ok=True)
    for t in tabs:
        dst = os.path.join(dest, "tab", os.path.basename(t))
        shutil.copyfile(t, dst)
        scritti.append(dst)

    os.makedirs(os.path.join(dest, "fig"), exist_ok=True)
    for pattern, stabile in FIGURES:
        cand = sorted(glob.glob(os.path.join(fig_dir, pattern)),
                      key=os.path.getmtime, reverse=True)
        if not cand:
            if verbose:
                print(f"  [figura assente: {pattern} — resta quella precedente]",
                      file=sys.stderr)
            continue
        if len(cand) > 1 and verbose:
            print(f"  [{pattern}: {len(cand)} candidati, prendo il piu' recente "
                  f"({os.path.basename(cand[0])})]", file=sys.stderr)
        dst = os.path.join(dest, "fig", stabile)
        shutil.copyfile(cand[0], dst)
        scritti.append(dst)
        # \graphicspath elenca Images/ PRIMA di Metrics/fig/, quindi un file con
        # lo stesso nome in Images/ vince e la figura generata non compare mai.
        # E' successo con fig_barrier_shape.pdf, residuo del report precedente:
        # il documento compilava pulito mostrando la figura sbagliata, con
        # parametri diversi da quelli descritti nel testo due righe sopra. Un
        # errore che non si annuncia va cercato, non aspettato.
        ombra = os.path.join(parent, "Images", stabile)
        if os.path.exists(ombra) and verbose:
            print(f"  [ATTENZIONE: {stabile} e' ombreggiata da {ombra} — "
                  f"Images/ precede Metrics/fig/ in \\graphicspath, quindi il "
                  f"report userebbe QUELLA. Rimuoverla.]", file=sys.stderr)

    return scritti


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=os.path.join(_HERE, "out", "tex"))
    ap.add_argument("--fig", default=os.path.join(_HERE, "out"))
    ap.add_argument("--dest", default=DEST_DEFAULT)
    args = ap.parse_args()
    scritti = sync(args.tex, args.fig, args.dest)
    print(f"aggiornati {len(scritti)} file in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
