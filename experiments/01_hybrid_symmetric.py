"""V14: due XGBoost, media 50/50. Importare il file non avvia il training."""


import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


CONFIGS = {'hybrid_d6': {'view': 'hybrid',
               'n_estimators': 700,
               'max_depth': 6,
               'learning_rate': 0.025,
               'min_child_weight': 20,
               'reg_lambda': 20,
               'subsample': 0.85,
               'colsample_bytree': 0.8},
 'symmetric_d4': {'view': 'symmetric',
                  'n_estimators': 600,
                  'max_depth': 4,
                  'learning_rate': 0.035,
                  'min_child_weight': 20,
                  'reg_lambda': 20,
                  'subsample': 0.9,
                  'colsample_bytree': 0.85}}

PESI = {'hybrid_d6': 0.5, 'symmetric_d4': 0.5}
SOGLIA = 0.5
CSV_NAME = 'submission_finale_1_hybrid_symmetric.csv'


def load_battles(path, has_labels=True):
    """Legge una battaglia per riga e controlla i campi necessari."""
    battles = []
    identifiers = set()
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            battle = json.loads(line)
            identifier = battle['battle_id']
            if identifier in identifiers:
                raise ValueError(f'battle_id duplicato alla riga {line_number}: {identifier}')
            identifiers.add(identifier)
            turns = [event['turn'] for event in battle['battle_timeline']]
            if turns != list(range(1, 31)):
                raise ValueError(f'La battaglia {identifier} non contiene i turni da 1 a 30.')
            names = [member['name'] for member in battle['p1_team_details']]
            if len(names) != 6 or len(set(names)) != 6:
                raise ValueError(f'Squadra non valida nella battaglia {identifier}.')
            if has_labels and battle.get('player_won') not in (0, 1):
                raise ValueError(f'Etichetta non valida nella battaglia {identifier}.')
            if not has_labels and 'player_won' in battle:
                raise ValueError('Il file di test non deve contenere le etichette.')
            battles.append(battle)
    if not battles:
        raise ValueError(f'Il file {path} è vuoto.')
    return battles


def battle_fingerprint(battle):
    """Confronta il contenuto, escludendo identificativo ed etichetta."""
    content = {}
    for name, value in battle.items():
        if name not in ('battle_id', 'player_won'):
            content[name] = value
    encoded = json.dumps(content, sort_keys=True, separators=(',', ':')).encode()
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
            if first_battle['player_won'] != battle['player_won']:
                raise ValueError('Due battaglie identiche hanno etichette diverse.')
            removed_rows.append({'removed_battle_id': battle['battle_id'], 'kept_battle_id': first_battle['battle_id']})
    return (unique_battles, removed_rows)


STATISTICS = ('base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe')
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe')
STATUSES = ('nostatus', 'brn', 'frz', 'par', 'psn', 'slp', 'tox', 'fnt', 'unknown')


def reconstruct_battle(battle, stop_turn=30):
    states = {'p1': {}, 'p2': {}}
    active_names = {'p1': None, 'p2': None}
    for event in battle['battle_timeline']:
        if event['turn'] > stop_turn:
            break
        for player in ('p1', 'p2'):
            observed = event.get(player + '_pokemon_state')
            if not observed or not observed.get('name'):
                continue
            name = observed['name']
            active_names[player] = name
            if name not in states[player]:
                states[player][name] = {'hp_pct': numpy.nan, 'hp_turn': None, 'status': 'unknown', 'status_turn': None, 'fainted': False}
            state = states[player][name]
            state['last_seen_turn'] = event['turn']
            state['boosts'] = observed.get('boosts') or {}
            state['effects'] = observed.get('effects')
            health = observed.get('hp_pct')
            if health is not None:
                if not 0 <= health <= 1:
                    raise ValueError('Gli HP osservati devono essere compresi tra 0 e 1.')
                state['hp_pct'] = float(health)
                state['hp_turn'] = event['turn']
            if observed.get('status') is not None:
                state['status'] = observed['status']
                state['status_turn'] = event['turn']
            if health == 0 or observed.get('status') == 'fnt':
                state['fainted'] = True
            if state['fainted']:
                state['hp_pct'] = 0.0
                state['status'] = 'fnt'
    return (states, active_names)


def add_summary(features, prefix, values):
    available = []
    for value in values:
        if value is not None and numpy.isfinite(value):
            available.append(float(value))
    features[prefix + '_count'] = len(available)
    features[prefix + '_sum'] = sum(available) if available else numpy.nan
    features[prefix + '_mean'] = float(numpy.mean(available)) if available else numpy.nan
    features[prefix + '_min'] = min(available) if available else numpy.nan
    features[prefix + '_max'] = max(available) if available else numpy.nan


def summarize_player(states, stop_turn):
    summary = {}
    observed = list(states.values())
    summary['observed_pokemon'] = len(observed)
    summary['unobserved_pokemon'] = 6 - len(observed)
    summary['fainted_pokemon'] = sum((state['fainted'] for state in observed))
    summary['remaining_pokemon'] = 6 - summary['fainted_pokemon']
    add_summary(summary, 'observed_health', [state['hp_pct'] for state in observed])
    health_ages = []
    for state in observed:
        if state['hp_turn'] is not None:
            health_ages.append(stop_turn - state['hp_turn'])
    add_summary(summary, 'health_observation_age', health_ages)
    summary['health_unknown_count'] = sum((not numpy.isfinite(state['hp_pct']) for state in observed))
    summary['health_under_25_percent'] = sum((0 < state['hp_pct'] <= 0.25 for state in observed))
    summary['health_under_50_percent'] = sum((0 < state['hp_pct'] <= 0.5 for state in observed))
    for status in STATUSES:
        summary['status_' + status] = sum((state['status'] == status for state in observed))
    return summary


def extract_base_features(battle, states, active_names):
    """Usa soltanto le informazioni disponibili nella singola battaglia."""
    features = {}
    summaries = {}
    for player in ('p1', 'p2'):
        summaries[player] = summarize_player(states[player], 30)
        for name, value in summaries[player].items():
            features[player + '_' + name] = value
        active_name = active_names[player]
        active = states[player].get(active_name, {})
        prefix = player + '_active_'
        features[prefix + 'name'] = active_name or 'unknown'
        features[prefix + 'health'] = active.get('hp_pct', numpy.nan)
        features[prefix + 'status'] = active.get('status', 'unknown')
        for statistic in BOOSTS:
            features[prefix + 'boost_' + statistic] = active.get('boosts', {}).get(statistic, numpy.nan)
        effects = active.get('effects')
        features[prefix + 'effects_unknown'] = int(effects is None)
        if isinstance(effects, str):
            effects = [effects]
        for effect in effects or []:
            if effect != 'noeffect':
                features[prefix + 'effect_' + effect] = 1
    for name in summaries['p1']:
        features['difference_' + name] = summaries['p1'][name] - summaries['p2'][name]
    for statistic in STATISTICS + ('level',):
        values = [member.get(statistic) for member in battle['p1_team_details']]
        add_summary(features, 'p1_initial_team_' + statistic, values)
        features['p2_initial_lead_' + statistic] = battle['p2_lead_details'].get(statistic, numpy.nan)
    for member in battle['p1_team_details']:
        for name in ['species_' + member['name'], 'item_' + member.get('item', 'unknown')]:
            key = 'p1_initial_team_' + name
            features[key] = features.get(key, 0) + 1
        for pokemon_type in member.get('types', []):
            key = 'p1_initial_team_type_' + pokemon_type
            features[key] = features.get(key, 0) + 1
    features['p2_initial_lead_name'] = battle['p2_lead_details']['name']
    for pokemon_type in battle['p2_lead_details'].get('types', []):
        features['p2_initial_lead_type_' + pokemon_type] = 1
    return features


def extract_resource_features(battle, states, active_names):
    """Salute dei singoli membri e risorse osservate, senza inventare HP."""
    features = {}
    for player in ('p1', 'p2'):
        known_names = list(states[player])
        if player == 'p1':
            initial_members = battle['p1_team_details']
        else:
            initial_members = [battle['p2_lead_details']]
        for member in initial_members:
            if member['name'] not in known_names:
                known_names.append(member['name'])
        for name in known_names:
            state = states[player].get(name, {})
            prefix = 'member_' + player + '_' + name + '_'
            health = state.get('hp_pct', numpy.nan)
            health_turn = state.get('hp_turn')
            features[prefix + 'known_member'] = 1
            features[prefix + 'observed'] = int(name in states[player])
            features[prefix + 'health_known'] = int(numpy.isfinite(health))
            features[prefix + 'health'] = health
            features[prefix + 'health_age'] = 30 - health_turn if health_turn is not None else numpy.nan
            features[prefix + 'fainted'] = int(state.get('fainted', False))
        alive_states = [state for state in states[player].values() if not state['fainted']]
        alive_health = [state['hp_pct'] for state in alive_states if numpy.isfinite(state['hp_pct'])]
        prefix = 'resource_' + player + '_'
        add_summary(features, prefix + 'observed_alive_health', alive_health)
        features[prefix + 'observed_alive_with_unknown_health'] = len(alive_states) - len(alive_health)
        sorted_health = sorted(alive_health, reverse=True)
        for rank in (1, 2, 3):
            features[prefix + 'largest_observed_health_' + str(rank)] = sorted_health[rank - 1] if len(sorted_health) >= rank else numpy.nan
        active = states[player].get(active_names[player], {})
        health = active.get('hp_pct', numpy.nan)
        features[prefix + 'active_fainted'] = int(active.get('fainted', False))
        for statistic in BOOSTS:
            boost = active.get('boosts', {}).get(statistic)
            if boost is None:
                boost = numpy.nan
            features[prefix + 'active_health_times_boost_' + statistic] = health * boost
    team = battle['p1_team_details']
    remaining = []
    for member in team:
        state = states['p1'].get(member['name'], {})
        if not state.get('fainted', False):
            remaining.append(member)
    for statistic in STATISTICS:
        values = [member[statistic] for member in remaining]
        prefix = 'resource_p1_remaining_' + statistic
        add_summary(features, prefix, values)
        total = sum((member[statistic] for member in team))
        features[prefix + '_retained_fraction'] = sum(values) / total if total > 0 else numpy.nan
        weighted_values = []
        for member in remaining:
            health = states['p1'].get(member['name'], {}).get('hp_pct', numpy.nan)
            if numpy.isfinite(health):
                weighted_values.append(member[statistic] * health)
        features[prefix + '_observed_health_weighted_sum'] = sum(weighted_values) if weighted_values else numpy.nan
    return features


def extract_temporal_features(battle, final_states):
    """Andamento osservato: finestre temporali e conteggi delle mosse disponibili."""
    features = {}
    for stop_turn in (10, 20, 30):
        if stop_turn == 30:
            snapshot = final_states
        else:
            snapshot, active_names = reconstruct_battle(battle, stop_turn)
        summaries = {}
        for player in ('p1', 'p2'):
            summary = summarize_player(snapshot[player], stop_turn)
            summaries[player] = summary
            for name in ('observed_pokemon', 'fainted_pokemon', 'observed_health_sum', 'observed_health_count', 'observed_health_mean', 'health_under_25_percent'):
                if stop_turn != 30:
                    features['time_' + player + '_turn_' + str(stop_turn) + '_' + name] = summary[name]
        if stop_turn != 30:
            for name in ('fainted_pokemon', 'observed_health_sum', 'observed_pokemon'):
                features['time_difference_turn_' + str(stop_turn) + '_' + name] = summaries['p1'][name] - summaries['p2'][name]
    for player in ('p1', 'p2'):
        previous_name = None
        previous_health = None
        fainted_names = set()
        windows = {}
        for window in (1, 2, 3):
            windows[window] = {'switches': 0, 'new_fainted': 0, 'health_increase': 0.0, 'health_decrease': 0.0, 'missing_moves': 0, 'physical': 0, 'special': 0, 'status': 0, 'powers': []}
        for event in battle['battle_timeline']:
            turn = event['turn']
            if not 1 <= turn <= 30:
                continue
            window = (turn - 1) // 10 + 1
            counters = windows[window]
            state = event.get(player + '_pokemon_state') or {}
            name = state.get('name')
            health = state.get('hp_pct')
            if name:
                if previous_name is not None and previous_name != name:
                    counters['switches'] += 1
                if name == previous_name and health is not None and (previous_health is not None):
                    change = health - previous_health
                    counters['health_increase'] += max(change, 0.0)
                    counters['health_decrease'] += max(-change, 0.0)
                is_fainted = health == 0 or state.get('status') == 'fnt'
                if is_fainted and name not in fainted_names:
                    counters['new_fainted'] += 1
                    fainted_names.add(name)
            previous_name = name
            previous_health = health
            move = event.get(player + '_move_details')
            if move is None:
                counters['missing_moves'] += 1
                continue
            category = move.get('category')
            if category in ('physical', 'special', 'status'):
                counters[category] += 1
            if category in ('physical', 'special'):
                counters['powers'].append(move.get('base_power'))
            for field in ('name', 'type', 'category'):
                key = 'move_' + player + '_' + field + '_' + str(move.get(field, 'unknown'))
                features[key] = features.get(key, 0) + 1
        for window, counters in windows.items():
            prefix = 'time_' + player + '_window_' + str(window) + '_'
            for name, value in counters.items():
                if name != 'powers':
                    features[prefix + name] = value
            add_summary(features, prefix + 'offensive_power', counters['powers'])
    return features


def extract_features(battle):
    states, active_names = reconstruct_battle(battle)
    features = extract_base_features(battle, states, active_names)
    features.update(extract_resource_features(battle, states, active_names))
    features.update(extract_temporal_features(battle, states))
    return features


TYPES = ('normal','fire','water','electric','grass','ice','fighting','poison',
         'ground','flying','psychic','bug','rock','ghost','dragon','dark','steel')
HEAL = {'rest','recover','softboiled','milkdrink','synthesis','moonlight','morningsun'}


def build_catalog(battles):
    records = defaultdict(list)
    for b in battles:
        for m in b['p1_team_details'] + [b['p2_lead_details']]:
            records[m['name']].append(m)
    catalog = {}
    for name, members in records.items():
        catalog[name] = {k: float(np.median([m[k] for m in members]))
                         for k in STATISTICS + ('level',)}
        catalog[name]['types'] = members[0]['types']
    return catalog


def load_chart(path):
    with open(path) as f:
        return {(r['attacking_type'], r['defending_type']): float(r['multiplier'])
                for r in csv.DictReader(f)}


def effectiveness(t, defender, chart):
    return float(np.prod([chart.get((t, d), 1.) for d in defender.get('types', [])]))


def summary(f, prefix, values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    for k, v in [('sum', a.sum()), ('mean', a.mean() if len(a) else np.nan),
                 ('max', a.max() if len(a) else np.nan),
                 ('min', a.min() if len(a) else np.nan)]:
        f[prefix+'_'+k] = float(v)


def boost_multiplier(b):
    return (2+b)/2 if b >= 0 else 2/(2-b)


def extract(b, catalog, chart, identity=False, observed_only=False):
    f = {}
    states, active = reconstruct_battle(b)
    own = {m['name']: m for m in b['p1_team_details'] + [b['p2_lead_details']]}
    info = {**catalog, **own}
    moves = {'p1': defaultdict(dict), 'p2': defaultdict(dict)}
    for e in b['battle_timeline']:
        if not 1 <= e['turn'] <= 30:
            continue
        for p in ('p1','p2'):
            s, m = e.get(p+'_pokemon_state') or {}, e.get(p+'_move_details')
            if s.get('name') and m:
                moves[p][s['name']][m['name']] = m

    resources = {}
    for p in ('p1','p2'):
        ss = states[p]
        names = sorted(set(ss) if observed_only else set(ss) | (
            set(m['name'] for m in b['p1_team_details']) if p == 'p1'
            else {b['p2_lead_details']['name']}))
        rows = []
        for name in names:
            s = ss.get(name, {})
            hp = s.get('hp_pct', np.nan)
            dead = s.get('fainted', False)
            # Estimate only: missing/unseen HP is also represented separately.
            h = 0. if dead else (hp if np.isfinite(hp) else 1.)
            m = info.get(name, {})
            status = s.get('status','unknown')
            revealed = moves[p][name]
            row = {'hp': h, 'dead': int(dead), 'unknown': int(not np.isfinite(hp)),
                   'age': 30-s.get('hp_turn',0), 'observed': int(name in ss),
                   'heal': int(bool(HEAL & revealed.keys())),
                   'resttalk': int('rest' in revealed and 'sleeptalk' in revealed),
                   'explosion': int(bool({'explosion','selfdestruct'} & revealed.keys())),
                   'status_load': float(status not in ('nostatus','unknown','fnt')) * h,
                   'speed': m.get('base_spe', np.nan),
                   'stats_known': int(bool(m))}
            for st in STATISTICS:
                row[st] = m.get(st,np.nan)
                row[st+'_hp'] = m.get(st,np.nan) * h
                row[st+'_alive'] = m.get(st,np.nan) * (h > 0)
            # Approximate resource scores, not an exact damage simulator.
            hpstat = m.get('base_hp',np.nan)
            row['physical_bulk'] = h*(2*hpstat+203)*(2*m.get('base_def',np.nan)+98)
            row['special_bulk'] = h*(2*hpstat+203)*(2*m.get('base_spd',np.nan)+98)
            for st in ('brn','par','psn','tox','slp','frz'):
                row['status_'+st] = float(status == st)*(h > 0)
            rows.append((name,row))
            if identity:
                pref = p+'_species_'+name+'_'
                for k in ('hp','dead','observed','heal','status_load'):
                    f[pref+k] = row[k]
        hidden = max(0,6-len(names))
        f[p+'_hidden'] = hidden
        f[p+'_hp_estimate'] = sum(r['hp'] for _,r in rows)+hidden
        f[p+'_hp_observed'] = sum(s['hp_pct'] for s in ss.values() if np.isfinite(s['hp_pct']))
        f[p+'_dead'] = sum(r['dead'] for _,r in rows)
        f[p+'_remaining'] = 6-f[p+'_dead']
        f[p+'_hp_per_remaining'] = f[p+'_hp_estimate']/max(1,f[p+'_remaining'])
        f[p+'_unknown_hp'] = sum(r['unknown'] for _,r in rows)+hidden
        f[p+'_active_name'] = active[p] or 'unknown'
        for k in rows[0][1]:
            summary(f,p+'_team_'+k,[r[k] for _,r in rows])
        for h in (.1,.25,.5,.75,.99):
            f[p+'_hp_over_'+str(h)] = sum(r['hp'] > h for _,r in rows)
        for rank, (_,r) in enumerate(sorted(rows,key=lambda z:(-z[1]['hp'],z[0]))):
            if rank >= 6:
                break
            for k in ('hp','status_load','speed','heal','resttalk','explosion') + STATISTICS:
                f[p+'_hp_rank_'+str(rank)+'_'+k] = r[k]
        # Current combatant: stats + effective boosts + revealed utility.
        a = ss.get(active[p],{})
        m = info.get(active[p],{})
        hp = a.get('hp_pct',np.nan)
        f[p+'_active_hp'] = hp
        f[p+'_active_status'] = a.get('status','unknown')
        f[p+'_active_age'] = 30-a.get('hp_turn',0)
        for st in STATISTICS + ('level',):
            f[p+'_active_'+st] = m.get(st,np.nan)
        for st in BOOSTS:
            boost = a.get('boosts',{}).get(st,0)
            value = (2*m.get('base_'+st,np.nan)+98)*boost_multiplier(boost)
            if st == 'spe' and a.get('status') == 'par':
                value *= .25
            if st == 'atk' and a.get('status') == 'brn':
                value *= .5
            f[p+'_active_boost_'+st] = boost
            f[p+'_active_effective_'+st] = value
            f[p+'_active_hp_boost_'+st] = hp*boost
        am = moves[p][active[p]]
        for k in ('rest','sleeptalk','curse','bellydrum','swordsdance','batonpass',
                  'toxic','spikes','rapidspin','explosion','selfdestruct','roar','whirlwind'):
            f[p+'_active_has_'+k] = int(k in am)
            f[p+'_team_has_'+k] = sum(k in moves[p][name] and not ss.get(name,{}).get('fainted',False)
                                       for name in names)
        for t in TYPES:
            f[p+'_type_hp_'+t] = sum(r['hp'] for name,r in rows if t in info.get(name,{}).get('types',[]))
        resources[p] = rows

    # Approximate attacking opportunities using only revealed moves and species types.
    for p,q in (('p1','p2'),('p2','p1')):
        am = info.get(active[p],{})
        dm = info.get(active[q],{})
        available = list(moves[p][active[p]].values())
        damage = []
        for move in available:
            if move.get('category') not in ('physical','special'):
                continue
            atk, defense = ('atk','def') if move['category'] == 'physical' else ('spa','spd')
            stab = 1.5 if move['type'] in am.get('types',[]) else 1.
            value = move.get('base_power',0)*effectiveness(move['type'],dm,chart)*stab
            value *= f[p+'_active_effective_'+atk]/max(1,f[q+'_active_effective_'+defense])
            value *= move.get('accuracy',1.) or 1.
            damage.append(value)
        summary(f,p+'_active_damage_proxy',damage)
        f[p+'_active_stab_effectiveness'] = max([effectiveness(t,dm,chart) for t in am.get('types',[]) if t != 'notype'] or [np.nan])
        matchup = []
        for name,r in resources[p]:
            if r['hp'] <= 0:
                continue
            m = info.get(name,{})
            attack_types = [v['type'] for v in moves[p][name].values() if v['category'] in ('physical','special')]
            attack_types += [t for t in m.get('types',[]) if t != 'notype']
            matchup.append(max([effectiveness(t,dm,chart) for t in attack_types] or [np.nan]))
        summary(f,p+'_team_vs_active',matchup)

    # Accumulated resources at multiple time horizons, and recent action pressure.
    for stop in (5,10,15,20,25,28,30):
        ss, aa = reconstruct_battle(b,stop)
        for p in ('p1','p2'):
            vals = list(ss[p].values())
            pref = p+'_t'+str(stop)+'_'
            f[pref+'dead'] = sum(s['fainted'] for s in vals)
            f[pref+'revealed'] = len(vals)
            f[pref+'hp_estimate'] = 6-len(vals)+sum(s['hp_pct'] if np.isfinite(s['hp_pct']) else 1 for s in vals)
            f[pref+'active_hp'] = ss[p].get(aa[p],{}).get('hp_pct',np.nan)
        for k in ('dead','hp_estimate','revealed','active_hp'):
            f['diff_t'+str(stop)+'_'+k] = f['p1_t'+str(stop)+'_'+k]-f['p2_t'+str(stop)+'_'+k]
    for p in ('p1','p2'):
        for start in (1,11,21,26):
            c = defaultdict(float)
            prev = None
            for e in b['battle_timeline']:
                if e['turn'] > 30:
                    break
                s = e.get(p+'_pokemon_state') or {}
                m = e.get(p+'_move_details')
                if e['turn'] >= start:
                    c['switches'] += int(prev is not None and s.get('name') != prev.get('name'))
                    if prev and s.get('name') == prev.get('name') and s.get('hp_pct') is not None and prev.get('hp_pct') is not None:
                        delta = s['hp_pct']-prev['hp_pct']
                        c['healing'] += max(0,delta)
                        c['loss'] += max(0,-delta)
                    if m:
                        c['move_'+m['name']] += 1
                        c['category_'+m['category']] += 1
                    else:
                        c['missing_moves'] += 1
                prev = s
            pref = p+'_since'+str(start)+'_'
            for k in ('switches','healing','loss','missing_moves','category_status','category_physical','category_special'):
                f[pref+k] = c[k]
            for move in HEAL | {'sleeptalk','spikes','rapidspin','curse','toxic','explosion','selfdestruct','batonpass'}:
                f[pref+'move_'+move] = c['move_'+move]
    # Explicit differences spare trees the need to rediscover basic comparisons.
    for key in list(f):
        if key.startswith('p1_') and isinstance(f[key],(int,float,np.number)):
            other = 'p2_'+key[3:]
            if other in f:
                f['diff_'+key[3:]] = f[key]-f[other]
    return f


def mirror(row):
    """Swap player perspectives of an observed-only row, without labels."""
    mirrored = {}
    for k,v in row.items():
        if k.startswith('p1_'):
            mirrored['p2_'+k[3:]] = v
        elif k.startswith('p2_'):
            mirrored['p1_'+k[3:]] = v
        elif k.startswith('diff_'):
            mirrored[k] = -v
        else:
            raise ValueError('Feature has no defined symmetry: '+k)
    return mirrored


def crea_matrici(battaglie_train, battaglie_eval, vista, tabella_tipi):
    # Catalogo e vocabolario sono costruiti esclusivamente sul training.
    catalogo = build_catalog(battaglie_train) if vista != "v8" else None

    def rappresenta(battaglia):
        if vista == "v8":
            riga = extract_features(battaglia)
        else:
            riga = extract(battaglia, catalogo, tabella_tipi,
                          identity=vista in ("hybrid", "symmetric"),
                          observed_only=vista == "symmetric")
            if vista == "hybrid":
                riga = {**extract_features(battaglia),
                        **{"new_"+k: v for k,v in riga.items()}}
        # Condivide i nomi delle caratteristiche per contenere la RAM.
        return {sys.intern(k):v for k,v in riga.items()}

    righe_train = [rappresenta(b) for b in battaglie_train]
    righe_eval = [rappresenta(b) for b in battaglie_eval]
    if vista == "symmetric":
        righe_train += [mirror(r) for r in righe_train]
        righe_eval += [mirror(r) for r in righe_eval]
    encoder = DictVectorizer(sparse=False, dtype=np.float32)
    X_train = encoder.fit_transform(righe_train)
    X_eval = encoder.transform(righe_eval)
    return X_train, X_eval, encoder, catalogo

def crea_modello(nome):
    parametri = {k:v for k,v in CONFIGS[nome].items() if k != "view"}
    return XGBClassifier(**parametri, objective="binary:logistic",
                         eval_metric="logloss", tree_method="hist",
                         random_state=42, n_jobs=2)

def allena_e_predici(nome, battaglie_train, battaglie_eval, y_train,
                     tabella_tipi, output_directory=None):
    vista = CONFIGS[nome]["view"]
    X_train, X_eval, encoder, catalogo = crea_matrici(
        battaglie_train, battaglie_eval, vista, tabella_tipi)
    etichette = np.r_[y_train, 1-y_train] if vista == "symmetric" else y_train
    modello = crea_modello(nome)
    with threadpool_limits(limits=2):
        modello.fit(X_train, etichette)
        p_raw = modello.predict_proba(X_eval)[:, 1]
    if output_directory is not None:
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        modello.save_model(Path(output_directory)/f"modello_{nome}.ubj")
        joblib.dump({"encoder":encoder, "catalogo":catalogo},
                    Path(output_directory)/f"caratteristiche_{nome}.joblib", compress=3)
        ricaricato = crea_modello(nome)
        ricaricato.load_model(Path(output_directory)/f"modello_{nome}.ubj")
        with threadpool_limits(limits=2):
            np.testing.assert_allclose(ricaricato.predict_proba(X_eval)[:,1], p_raw,
                                       rtol=0, atol=1e-7)
    n = len(battaglie_eval)
    # Per ogni battaglia simmetrica media la previsione normale e quella invertita.
    p = (p_raw[:n] + 1-p_raw[n:])/2 if vista == "symmetric" else p_raw
    del X_train, X_eval, encoder, catalogo, modello
    gc.collect()
    return p


def cross_validate(train_battles, y, chart):
    """Five folds; catalog and encoder are fitted inside each training fold."""
    oof = {name: np.full(len(y), np.nan) for name in PESI}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        training = [train_battles[i] for i in tr]
        validation = [train_battles[i] for i in va]
        for name in PESI:
            oof[name][va] = allena_e_predici(name, training, validation, y[tr], chart)
        print(f"Fold {fold}/5 completato", flush=True)
    probabilities = sum(weight * oof[name] for name, weight in PESI.items())
    metrics = {
        "accuracy": float(accuracy_score(y, probabilities >= SOGLIA)),
        "log_loss": float(log_loss(y, probabilities)),
    }
    print("CV:", metrics, flush=True)
    return metrics


def save_submission(test_battles, sample, probabilities, output_directory):
    """Align predictions to the sample IDs, then write the submission."""
    test_ids = [battle["battle_id"] for battle in test_battles]
    if list(sample.columns) != ["battle_id", "player_won"]:
        raise ValueError("Colonne del sample_submission non valide.")
    if sample["battle_id"].isna().any() or not sample["battle_id"].is_unique:
        raise ValueError("Il sample contiene ID mancanti o duplicati.")
    if set(test_ids) != set(sample["battle_id"]):
        raise ValueError("Gli ID del test e del sample non coincidono.")
    frame = pd.DataFrame({"battle_id": test_ids, "probability": probabilities})
    frame = sample[["battle_id"]].merge(
        frame, on="battle_id", how="left", sort=False, validate="one_to_one")
    p = frame["probability"].to_numpy()
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Probabilita mancanti o fuori da [0, 1].")
    submission = sample.copy()
    submission["player_won"] = (p >= SOGLIA).astype(int)
    csv_path = Path(output_directory) / CSV_NAME
    submission.to_csv(csv_path, index=False)
    frame.to_csv(Path(output_directory) / "probabilita.csv", index=False)
    np.testing.assert_array_equal(pd.read_csv(csv_path)["player_won"], submission["player_won"])
    print("CSV creato:", csv_path, flush=True)
    print("Righe:", len(submission), "- positive:", int(submission.player_won.sum()))
    return csv_path


def run(data_directory, output_directory="outputs/01_hybrid_symmetric",
        run_cv=False, save_models=False):
    data_directory = Path(data_directory).expanduser()
    required_files = ("train.jsonl", "test.jsonl", "sample_submission.csv", "type_chart_gen2.csv")
    missing = [name for name in required_files if not (data_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"File mancanti in {data_directory}: {', '.join(missing)}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "threadpoolctl")}
    print("Versioni:", versions, flush=True)
    print("Dataset:", data_directory, flush=True)
    train_battles, duplicates = remove_duplicate_battles(
        load_battles(data_directory / "train.jsonl"))
    test_battles = load_battles(data_directory / "test.jsonl", has_labels=False)
    y = np.array([battle["player_won"] for battle in train_battles])
    sample = pd.read_csv(data_directory / "sample_submission.csv")
    chart = load_chart(data_directory / "type_chart_gen2.csv")
    print("Training unico:", len(train_battles), "- duplicati rimossi:", len(duplicates), flush=True)
    metrics = cross_validate(train_battles, y, chart) if run_cv else None
    probabilities = {}
    for name in PESI:
        print("Addestramento:", name, flush=True)
        probabilities[name] = allena_e_predici(
            name, train_battles, test_battles, y, chart,
            output_directory=output_directory if save_models else None).astype(np.float64)
    final = sum(weight * probabilities[name] for name, weight in PESI.items())
    csv_path = save_submission(test_battles, sample, final, output_directory)
    metadata = {
        "versions": versions, "configs": CONFIGS, "weights": PESI, "threshold": SOGLIA,
        "cv": metrics,
    }
    (output_directory / "run_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="V14: XGBoost ibrido + simmetrico (50/50).")
    parser.add_argument("--data-dir", type=Path, required=True, help="Cartella con i quattro file del dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/01_hybrid_symmetric"))
    parser.add_argument("--cv", action="store_true", help="Aggiunge la CV a cinque fold.")
    parser.add_argument("--save-models", action="store_true", help="Salva anche modelli ed encoder.")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.cv, args.save_models)


if __name__ == "__main__":
    main()

