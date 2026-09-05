# metrics — misura del problema di ottimizzazione

Strumenti offline che ri-risolvono e misurano lo stesso NLP che gira a bordo.
Ogni script documenta nella propria docstring che cosa calcola e come si lancia.

```bash
python3 metrics/test_fidelity.py        # il costo disegnato == il costo ottimizzato
python3 metrics/cost_field.py           # pannello 1: paesaggio di navigazione c(x,y)
python3 metrics/decision_plane.py       # pannello 2: spazio delle decisioni + iterati IPOPT
```

Non richiedono ROS: bastano `casadi`, `numpy`, `scipy`, `matplotlib`, `pyyaml`.
Le figure finiscono in `metrics/out/`.

## Numeri per il report

```bash
python3 metrics/make_results.py         # esegue tutte le misure e scrive tutto
python3 metrics/results_tex.py          # solo il LaTeX, da un results.json esistente
python3 metrics/results_tex.py --check  # verifica il LaTeX senza scriverlo
```

`make_results.py` produce cinque file dalla stessa campagna:

| file | a cosa serve |
|---|---|
| `out/results.json` | i numeri in forma strutturata, con la provenienza (commit, profilo, versioni) |
| `out/results.md` | gli stessi numeri da leggere in terminale |
| `out/tex/metrics_macros.tex` | una macro per ogni scalare — è il pezzo che rende automatico l'aggiornamento del report |
| `out/tex/metrics_body.tex` | sezioni e tabelle senza preambolo, da `\input{}` dentro `Report.tex` |
| `out/tex/metrics_standalone.tex` | wrapper per compilare le metriche da sole |

La regola che tiene insieme tutto: **nel report non si scrive un numero a mano**.
Si scrive `$\resOrderMidpoint$`, e il numero segue il codice. Una taratura
ritoccata e un `make_results.py` rieseguito aggiornano report e metriche
insieme, invece di farli divergere in silenzio — che è esattamente quello che
era successo a `snippets/nlp_structure.py`, rimasto a una taratura
superata.

L'ordine delle sezioni in `metrics_body.tex` segue il report, non il codice, e
ogni blocco dichiara con un `\resNote{...}` a quale sezione del report è
destinato. La tabella di corrispondenza in testa al documento è il piano di
integrazione.

### Script satellite

Non tutte le misure passano da `make_results.py`: questi scrivono un proprio
JSON in `metrics/out/`, che `results_tex.py` raccoglie **se lo trova**.

| script | JSON | sezione generata |
|---|---|---|
| `metrics/horizon_sweep.py` | `horizon_sweep.json` | orizzonte N × dt, fasce e insieme non dominato |
| `metrics/control_horizon.py` | `control_horizon.json` | orizzonte di controllo N_c, e perché non il move blocking |
| `metrics/shooting_compare.py` | `shooting_compare.json` | single contro multiple shooting |
| `metrics/solver_compare.py` | `solver_compare.json` | interior point contro active set |
| `metrics/robust_constraints.py` | `robust_constraints.json` | constraint tightening dal tubo misurato |
| `metrics/pareto_front.py` | `pareto_front.json` | fronte di Pareto multi-obiettivo |

L'assenza di un file salta la sola sezione; un cambio di schema la salta con un
avviso su stderr, senza far cadere il documento. Quindi si può rigenerare il
`.tex` in qualsiasi momento, anche mentre uno di questi script è in lavorazione.

### Cosa fa il generatore prima di scrivere

`results_tex.py` non si limita a formattare: rifiuta di scrivere un `.tex` rotto.
Controlla graffe e `\begin`/`\end` bilanciati, righe con un numero di celle
diverso dalle colonne del `tabular`, token matematici fuori da `$...$`, macro
usate nella prosa ma mai definite, e `\ref` senza `\label`. Sono i cinque modi
in cui un generatore di LaTeX sbaglia in silenzio.

Le affermazioni nella prosa sono **condizionate ai dati**: se la pendenza della
coda ℓ² non arriva a −1, se il fronte di Pareto non è informativo, se nello
sweep non c'è un orizzonte che degrada, il testo lo dice invece di riportare la
conclusione attesa. Aggiungere una bag nuova e rigenerare non può quindi
produrre un documento che afferma cose che i nuovi dati non sostengono.
