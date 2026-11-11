# Ricette Python e notebook Kaggle

Il notebook Kaggle scarica il codice da questo repository ed esegue `run_all.py`.
Ogni ricetta gira in un processo separato e salva i risultati nella propria cartella.

Al momento e disponibile **1 delle 3 ricette previste**:

| Script | Modello | CSV |
| --- | --- | --- |
| `experiments/01_hybrid_symmetric.py` | XGBoost ibrido + XGBoost simmetrico, probabilita mediate 50/50 | `submission_finale_1_hybrid_symmetric.csv` |

Le altre due ricette saranno convertite dai rispettivi notebook e aggiunte alla lista
`EXPERIMENTS` in `run_all.py`. Il notebook Kaggle non dovra cambiare.
I notebook e lo script gia presenti nel repository restano disponibili.

## Su Kaggle

1. Importa `notebooks/run_kaggle.ipynb` in Kaggle.
2. Aggiungi il dataset contenente `train.jsonl`, `test.jsonl`,
   `sample_submission.csv` e `type_chart_gen2.csv` nella stessa cartella.
3. Attiva **Internet** nelle impostazioni del notebook.
4. Il repository e privato: in **Add-ons > Secrets** aggiungi `GITHUB_TOKEN`,
   un token GitHub con accesso in lettura ai contenuti di questo repository,
   e abilitalo per il notebook. Il token non va scritto nelle celle.
5. Esegui tutte le celle. Se il dataset non viene trovato automaticamente,
   imposta `DATA_DIR` nella prima cella.

Gli output sono in `/kaggle/working/outputs/01_hybrid_symmetric/`.
La cella finale mostra i link ai CSV di submission, uno per ricetta.
Non combina le tre ricette in un ulteriore ensemble.
L'autenticazione e gia predisposta; per eseguire davvero il notebook serve il Secret
configurato nel tuo account Kaggle.

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

- `--output-dir PATH`: cartella dei risultati.
- `--cv`: aggiunge la validazione a cinque fold (seed 17).
- `--save-models`: salva modelli XGBoost ed encoder e verifica il caricamento.
- `--skip-data-hashes`: consente dati diversi dagli originali, ad esempio per una prova.
- `python run_all.py --list`: elenca le ricette senza addestrare.

Se `--data-dir` e omesso, la ricerca automatica controlla le cartelle locali,
`FDS_DATA_DIRECTORY` e `/kaggle/input`; piu dataset compatibili richiedono un percorso esplicito.

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
non costituisce una nuova valutazione. Gli hash dei dati originali sono controllati
per impostazione iniziale; quello della submission viene confrontato a fine esecuzione.
Versioni diverse delle librerie possono cambiare le previsioni. `run_info.json`
registra versioni effettive, configurazione e hash, senza promettere identita numerica
fra ambienti diversi. Il file `requirements.txt` preesistente non e stato modificato.

La verifica locale della conversione usa anche dati sintetici; il training completo
sui dati della competizione richiede i quattro file del dataset.
