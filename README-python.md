# Ricette Python e notebook Kaggle

Il notebook Kaggle scarica il codice pubblico con `!git` ed esegue `run_all.py` con `!python`.
Ogni ricetta gira in un processo separato e salva i risultati nella propria cartella.

Al momento e disponibile **1 delle 3 ricette previste**:

| Script | Modello | CSV |
| --- | --- | --- |
| `experiments/01_hybrid_symmetric.py` | XGBoost ibrido + XGBoost simmetrico, probabilita mediate 50/50 | `submission_finale_1_hybrid_symmetric.csv` |

Le altre due ricette saranno convertite dai rispettivi notebook e aggiunte alla lista
`EXPERIMENTS` in `run_all.py`. Il notebook Kaggle non dovra cambiare.

## Su Kaggle

1. Rendi pubblico il repository e importa `notebooks/run_kaggle.ipynb` in Kaggle,
   oppure copia le sue celle nel notebook che hai gia aperto.
2. Aggiungi il dataset contenente `train.jsonl`, `test.jsonl`,
   `sample_submission.csv` e `type_chart_gen2.csv` nella stessa cartella.
3. Attiva **Internet** nelle impostazioni del notebook.
4. Imposta `DATA_DIR` nella prima cella sulla cartella contenente i quattro file.
   Il percorso iniziale e `/kaggle/input/competitions/fds-pokemon-battles-prediction-2026`:
   controlla che corrisponda alla posizione dei file nel pannello Input.
5. Esegui tutte le celle. La prima esecuzione clona la repo, quelle successive
   aggiornano il codice con `git pull --ff-only`. Non servono token o GPU.

Gli output sono in `/kaggle/working/outputs/01_hybrid_symmetric/`.
La cella finale mostra i link ai CSV di submission, uno per ricetta.
Non combina le tre ricette in un ulteriore ensemble.
Le celle controllano l'esito dei comandi shell: un errore interrompe Run All.
Per aggiungere la CV o salvare i modelli, aggiungi `--cv` o `--save-models`
al comando `!python`.

## In locale

```bash
python -m pip install -r requirements-kaggle.txt
python run_all.py --data-dir /percorso/dataset
```

Per eseguire solo la prima ricetta:

```bash
python experiments/01_hybrid_symmetric.py --data-dir /percorso/dataset
```

Opzioni condivise:

- `--data-dir PATH`: cartella obbligatoria contenente i quattro file del dataset.
- `--output-dir PATH`: cartella dei risultati.
- `--cv`: aggiunge la validazione a cinque fold (seed 17).
- `--save-models`: salva modelli XGBoost ed encoder e verifica il caricamento.
- `python run_all.py --list`: elenca le ricette senza addestrare.

Gli script leggono soltanto la cartella indicata con `--data-dir`.
Se manca uno dei quattro file, si fermano con un messaggio esplicito.
`--list` funziona anche senza indicare il dataset.

## Cosa conserva la conversione

Le funzioni delle caratteristiche, la rimozione dei duplicati, i parametri XGBoost,
la simmetria, i seed, i due thread CPU, i pesi 50/50 e la soglia 0.5 provengono dal
notebook `01_V14_ibrido_simmetrico_069.ipynb`.
Catalogo e vocabolario vengono costruiti sul solo training, anche nella CV.
Importare lo script non legge dati e non avvia addestramenti.

Per impostazione iniziale si producono submission, probabilita e `run_info.json`;
CV e salvataggio dei modelli sono facoltativi. Gli ID seguono l'ordine del sample.
Il numero di righe e ricavato dal sample anziche essere fissato a 5000.

Lo score **0.69000 e storico**, comunicato nel notebook originale: la conversione
non costituisce una nuova valutazione. I confronti con gli SHA storici di dataset
e submission sono stati rimossi; resta l'hash delle battaglie per trovare i duplicati.
Versioni diverse delle librerie possono cambiare le previsioni. `run_info.json`
registra versioni effettive, configurazione ed eventuali metriche CV, senza promettere
identita numerica fra ambienti diversi.

La verifica locale della conversione usa anche dati sintetici; il training completo
sui dati della competizione richiede i quattro file del dataset.
