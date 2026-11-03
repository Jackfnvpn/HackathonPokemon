"""Un solo XGBoost: prepara i dati, valida il modello e crea una submission.

Importare questo modulo non avvia alcun addestramento.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


RANDOM_STATE = 42
MODEL_PARAMETERS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}
STATISTICS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe")
STATUSES = ("nostatus", "brn", "frz", "par", "psn", "slp", "tox", "fnt", "unknown")


# 1. Caricamento e rimozione delle copie esatte

def load_battles(path, has_labels=True):
    """Legge una battaglia per riga e controlla i campi necessari."""
    battles = []
    identifiers = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            battle = json.loads(line)
            identifier = battle["battle_id"]
            if identifier in identifiers:
                raise ValueError(f"battle_id duplicato alla riga {line_number}: {identifier}")
            identifiers.add(identifier)
            turns = [event["turn"] for event in battle["battle_timeline"]]
            if turns != list(range(1, 31)):
                raise ValueError(f"La battaglia {identifier} non contiene i turni da 1 a 30.")
            names = [member["name"] for member in battle["p1_team_details"]]
            if len(names) != 6 or len(set(names)) != 6:
                raise ValueError(f"Squadra non valida nella battaglia {identifier}.")
            if has_labels and battle.get("player_won") not in (0, 1):
                raise ValueError(f"Etichetta non valida nella battaglia {identifier}.")
            if not has_labels and "player_won" in battle:
                raise ValueError("Il file di test non deve contenere le etichette.")
            battles.append(battle)
    if not battles:
        raise ValueError(f"Il file {path} è vuoto.")
    return battles


def battle_fingerprint(battle):
    """Confronta il contenuto, escludendo identificativo ed etichetta."""
    content = {}
    for name, value in battle.items():
        if name not in ("battle_id", "player_won"):
            content[name] = value
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def remove_duplicate_battles(battles):
    unique_battles = []
    first_occurrences = {}
    removed_rows = []
    for battle in battles:
        fingerprint = battle_fingerprint(battle)
        if fingerprint not in first_occurrences:
            first_occurrences[fingerprint] = battle
            unique_battles.append(battle)
        else:
            first_battle = first_occurrences[fingerprint]
            if first_battle["player_won"] != battle["player_won"]:
                raise ValueError("Due battaglie identiche hanno etichette diverse.")
            removed_rows.append({"removed_battle_id": battle["battle_id"],
                                 "kept_battle_id": first_battle["battle_id"]})
    return unique_battles, removed_rows


# 2. Ultimo stato osservato di ogni Pokémon

def reconstruct_battle(battle, stop_turn=30):
    states = {"p1": {}, "p2": {}}
    active_names = {"p1": None, "p2": None}
    for event in battle["battle_timeline"]:
        if event["turn"] > stop_turn:
            break
        for player in ("p1", "p2"):
            observed = event.get(player + "_pokemon_state")
            if not observed or not observed.get("name"):
                continue
            name = observed["name"]
            active_names[player] = name
            if name not in states[player]:
                states[player][name] = {
                    "hp_pct": numpy.nan, "hp_turn": None,
                    "status": "unknown", "status_turn": None, "fainted": False,
                }
            state = states[player][name]
            state["last_seen_turn"] = event["turn"]
            state["boosts"] = observed.get("boosts") or {}
            state["effects"] = observed.get("effects")
            health = observed.get("hp_pct")
            if health is not None:
                if not 0 <= health <= 1:
                    raise ValueError("Gli HP osservati devono essere compresi tra 0 e 1.")
                state["hp_pct"] = float(health)
                state["hp_turn"] = event["turn"]
            if observed.get("status") is not None:
                state["status"] = observed["status"]
                state["status_turn"] = event["turn"]
            if health == 0 or observed.get("status") == "fnt":
                state["fainted"] = True
            # Un esausto conta una sola volta, anche se ricompare nel registro.
            if state["fainted"]:
                state["hp_pct"] = 0.0
                state["status"] = "fnt"
    return states, active_names


def add_summary(features, prefix, values):
    available = []
    for value in values:
        if value is not None and numpy.isfinite(value):
            available.append(float(value))
    features[prefix + "_count"] = len(available)
    features[prefix + "_sum"] = sum(available) if available else numpy.nan
    features[prefix + "_mean"] = float(numpy.mean(available)) if available else numpy.nan
    features[prefix + "_min"] = min(available) if available else numpy.nan
    features[prefix + "_max"] = max(available) if available else numpy.nan


def summarize_player(states, stop_turn):
    summary = {}
    observed = list(states.values())
    summary["observed_pokemon"] = len(observed)
    summary["unobserved_pokemon"] = 6 - len(observed)
    summary["fainted_pokemon"] = sum(state["fainted"] for state in observed)
    summary["remaining_pokemon"] = 6 - summary["fainted_pokemon"]
    add_summary(summary, "observed_health", [state["hp_pct"] for state in observed])
    health_ages = []
    for state in observed:
        if state["hp_turn"] is not None:
            health_ages.append(stop_turn - state["hp_turn"])
    add_summary(summary, "health_observation_age", health_ages)
    summary["health_unknown_count"] = sum(not numpy.isfinite(state["hp_pct"]) for state in observed)
    summary["health_under_25_percent"] = sum(0 < state["hp_pct"] <= 0.25 for state in observed)
    summary["health_under_50_percent"] = sum(0 < state["hp_pct"] <= 0.5 for state in observed)
    for status in STATUSES:
        summary["status_" + status] = sum(state["status"] == status for state in observed)
    return summary


# 3. Una riga di caratteristiche per ogni battaglia

def extract_features(battle):
    """Usa soltanto le informazioni disponibili nella singola battaglia."""
    states, active_names = reconstruct_battle(battle)
    features = {}
    summaries = {}
    for player in ("p1", "p2"):
        summaries[player] = summarize_player(states[player], 30)
        for name, value in summaries[player].items():
            features["" + player + "_" + name] = value
        active_name = active_names[player]
        active = states[player].get(active_name, {})
        prefix = "" + player + "_active_"
        features[prefix + "name"] = active_name or "unknown"
        features[prefix + "health"] = active.get("hp_pct", numpy.nan)
        features[prefix + "status"] = active.get("status", "unknown")
        for statistic in BOOSTS:
            features[prefix + "boost_" + statistic] = active.get("boosts", {}).get(statistic, numpy.nan)
        effects = active.get("effects")
        features[prefix + "effects_unknown"] = int(effects is None)
        if isinstance(effects, str):
            effects = [effects]
        for effect in effects or []:
            if effect != "noeffect":
                features[prefix + "effect_" + effect] = 1
    for name in summaries["p1"]:
        features["difference_" + name] = summaries["p1"][name] - summaries["p2"][name]
    for statistic in STATISTICS + ("level",):
        values = [member.get(statistic) for member in battle["p1_team_details"]]
        add_summary(features, "p1_initial_team_" + statistic, values)
        features["p2_initial_lead_" + statistic] = battle["p2_lead_details"].get(statistic, numpy.nan)
    for member in battle["p1_team_details"]:
        for name in ["species_" + member["name"], "item_" + member.get("item", "unknown")]:
            key = "p1_initial_team_" + name
            features[key] = features.get(key, 0) + 1
        for pokemon_type in member.get("types", []):
            key = "p1_initial_team_type_" + pokemon_type
            features[key] = features.get(key, 0) + 1
    features["p2_initial_lead_name"] = battle["p2_lead_details"]["name"]
    for pokemon_type in battle["p2_lead_details"].get("types", []):
        features["p2_initial_lead_type_" + pokemon_type] = 1
    return features


# 4. Un'unica architettura, con parametri fissi

def create_model(number_of_threads=4):
    """Crea il modello senza addestrarlo."""
    classifier = XGBClassifier(
        **MODEL_PARAMETERS,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        n_jobs=number_of_threads,
    )
    # La codifica impara le categorie soltanto dai dati passati a fit.
    # XGBoost gestisce direttamente i valori numerici mancanti (NaN).
    return Pipeline([
        ("encoding", DictVectorizer(sparse=False, dtype=numpy.float32)),
        ("classifier", classifier),
    ])


def validate_model(battles, output_directory, number_of_threads=4):
    """Addestra sul 80% e misura l'Accuracy sul 20% lasciato da parte."""
    feature_rows = [extract_features(battle) for battle in battles]
    labels = numpy.array([battle["player_won"] for battle in battles])
    training_indices, validation_indices = train_test_split(
        numpy.arange(len(battles)), test_size=0.2,
        stratify=labels, random_state=RANDOM_STATE,
    )
    training_rows = [feature_rows[index] for index in training_indices]
    validation_rows = [feature_rows[index] for index in validation_indices]
    model = create_model(number_of_threads)
    with threadpool_limits(limits=number_of_threads):
        model.fit(training_rows, labels[training_indices])
        probabilities = model.predict_proba(validation_rows)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(labels[validation_indices], predictions)),
        "confusion_matrix": confusion_matrix(
            labels[validation_indices], predictions, labels=[0, 1]
        ).tolist(),
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "training_battle_ids": [battles[index]["battle_id"] for index in training_indices],
        "validation_battle_ids": [battles[index]["battle_id"] for index in validation_indices],
        "parameters": dict(MODEL_PARAMETERS),
        "random_state": RANDOM_STATE,
        "threshold": 0.5,
        "versions": package_versions(),
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# 5. Modello finale e unico CSV delle previsioni sul test

def validate_submission(submission, sample_submission):
    if list(submission.columns) != ["battle_id", "player_won"]:
        raise ValueError("Le colonne della submission non sono corrette.")
    if len(submission) != len(sample_submission):
        raise ValueError("Manca una previsione per qualche battaglia.")
    if submission["battle_id"].duplicated().any():
        raise ValueError("La submission contiene identificativi duplicati.")
    if submission["battle_id"].tolist() != sample_submission["battle_id"].tolist():
        raise ValueError("Gli identificativi non seguono l'ordine del file di esempio.")
    if not pandas.api.types.is_integer_dtype(submission["player_won"]):
        raise ValueError("Le etichette devono essere numeri interi.")
    if not submission["player_won"].isin([0, 1]).all():
        raise ValueError("Le previsioni devono essere 0 oppure 1.")


def train_and_create_submission(training_battles, test_battles, sample_submission,
                                output_directory, number_of_threads=4):
    """Riaddestra lo stesso XGBoost su tutti gli esempi unici e salva submission.csv."""
    test_identifiers = [battle["battle_id"] for battle in test_battles]
    if list(sample_submission.columns) != ["battle_id", "player_won"]:
        raise ValueError("Il file di esempio non contiene le colonne previste.")
    if sample_submission["battle_id"].duplicated().any():
        raise ValueError("Il file di esempio contiene identificativi duplicati.")
    if len(set(test_identifiers)) != len(test_identifiers):
        raise ValueError("Il test contiene identificativi duplicati.")
    if set(test_identifiers) != set(sample_submission["battle_id"]):
        raise ValueError("Test e file di esempio contengono identificativi diversi.")
    training_rows = [extract_features(battle) for battle in training_battles]
    test_rows = [extract_features(battle) for battle in test_battles]
    labels = [battle["player_won"] for battle in training_battles]
    model = create_model(number_of_threads)
    with threadpool_limits(limits=number_of_threads):
        model.fit(training_rows, labels)
        probabilities = model.predict_proba(test_rows)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    predictions_by_identifier = dict(zip(test_identifiers, predictions))
    submission = sample_submission.copy()
    submission["player_won"] = submission["battle_id"].map(predictions_by_identifier).astype(int)
    validate_submission(submission, sample_submission)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_directory / "submission.csv", index=False)
    information = {
        "model": "XGBoost", "parameters": dict(MODEL_PARAMETERS),
        "training_rows": len(training_battles), "threshold": 0.5,
        "random_state": RANDOM_STATE, "versions": package_versions(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    joblib.dump({"model": model, "information": information},
                output_directory / "model.joblib", compress=3)
    (output_directory / "model_information.json").write_text(
        json.dumps(information, indent=2), encoding="utf-8"
    )
    return submission


def predict_with_saved_model(model_path, battles):
    saved = joblib.load(model_path)
    rows = [extract_features(battle) for battle in battles]
    probabilities = saved["model"].predict_proba(rows)[:, 1]
    return (probabilities >= saved["information"]["threshold"]).astype(int)


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "threadpoolctl"):
        versions[name] = importlib.metadata.version(name)
    return versions

