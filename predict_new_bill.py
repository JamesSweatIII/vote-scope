import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

# --------- Load artifacts ---------
def load_artifacts(path_prefix: str = "legislator_vote_models"):
    models = joblib.load(f"{path_prefix}.models.joblib")       # dict: legislator_id -> sklearn model
    meta   = joblib.load(f"{path_prefix}.meta.joblib")         # dict: legislator_id -> metrics
    with open(f"{path_prefix}.emb_model.txt") as f:
        emb_model_name = f.read().strip()
    emb_model = SentenceTransformer(emb_model_name)
    return models, meta, emb_model

# --------- Build roster from your voting_df ---------
def build_roster_from_votes(voting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per legislator with (legislator_id, name, party, state, chamber).
    'chamber' is inferred heuristically as the mode of 'origin_chamber' they appear with.
    If origin_chamber is missing or mixed, 'Unknown' is used.
    """
    base = (voting_df[['legislator_id', 'name', 'party', 'state']]
            .drop_duplicates(subset=['legislator_id'])
            .copy())

    # Best-effort chamber inference from your data (origin_chamber = bill origin; not perfect but works if dataset is chamber-specific)
    ch_map = (voting_df[['legislator_id', 'origin_chamber']]
              .dropna()
              .value_counts()
              .reset_index(name='n'))
    # pick most frequent chamber per legislator
    ch_mode = (ch_map.sort_values(['legislator_id', 'n'], ascending=[True, False])
                     .drop_duplicates(subset=['legislator_id'])
                     .rename(columns={'origin_chamber': 'chamber'})[['legislator_id','chamber']])

    roster = base.merge(ch_mode, on='legislator_id', how='left')
    roster['chamber'] = roster['chamber'].fillna('Unknown')
    return roster

# --------- Core prediction helpers ---------
def _compose_text(summary: str, title: str | None = None, bill_type_expanded: str | None = None) -> str:
    parts = [summary.strip()]
    if title and isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if bill_type_expanded and isinstance(bill_type_expanded, str) and bill_type_expanded.strip():
        parts.append(bill_type_expanded.strip())
    return " ".join(parts)

def predict_prob_yes_for_all(
    emb_model: SentenceTransformer,
    models: dict,
    text: str,
) -> pd.DataFrame:
    vec = emb_model.encode([text], normalize_embeddings=True)
    rows = []
    for leg_id, clf in models.items():
        p_yes = float(clf.predict_proba(vec)[0, 1])
        rows.append((leg_id, p_yes))
    return pd.DataFrame(rows, columns=['legislator_id', 'prob_yes'])

def majority_needed(n_members: int) -> int:
    return (n_members // 2) + 1

def chamber_pass_estimate(
    pred_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    chamber: str,
    threshold: float = 0.5,
) -> dict:
    dfc = roster_df.copy()
    EXCLUDE_NAMES = {"Hernandez","King-Hinds","Moylan","Norton","Plaskett","Radewagen","Fine","Patronis"}
    dfc = dfc[~dfc["name"].isin(EXCLUDE_NAMES)] 
    dfc = dfc[dfc['chamber'].str.lower() == chamber.lower()]
    dfc = dfc.merge(pred_df, on='legislator_id', how='left')

    # Handle legislators with no model (e.g., not enough training data)
    dfc['prob_yes'] = dfc['prob_yes'].fillna(0.5)  # neutral fallback; or 0.0 if you prefer

    dfc['pred_vote'] = (dfc['prob_yes'] >= threshold).astype(int)
    n_members = len(dfc)
    exp_yes = float(dfc['prob_yes'].sum())
    maj = majority_needed(n_members)
    would_pass = (exp_yes >= maj)

    return {
        'chamber': chamber,
        'members': n_members,
        'simple_majority_needed': maj,
        'expected_yes_votes': exp_yes,
        'would_pass_simple_majority': bool(would_pass),
        'details': dfc.sort_values('prob_yes', ascending=False).reset_index(drop=True)
    }

# --------- Public API ---------
def evaluate_new_bill(
    summary: str,
    voting_df: pd.DataFrame,
    title: str | None = None,
    bill_type_expanded: str | None = None,
    models_path_prefix: str = "legislator_vote_models",
    vote_threshold: float = 0.5,
    chambers_to_check: list[str] = ("House"),
    congress: int = 119,
):
    """
    Returns:
      results: dict with per-chamber pass estimates and a combined per-legislator DataFrame.
    """
    # load models + embedder
    models, meta, emb_model = load_artifacts(models_path_prefix)

    # compose text (you can pass just summary; title/type are optional)
    text = _compose_text(summary, title, bill_type_expanded)

    # per-legislator probs
    pred = predict_prob_yes_for_all(emb_model, models, text)

    # join identity info
    voting_df = voting_df[voting_df['congress'] == congress]
    roster = build_roster_from_votes(voting_df)
    pred_all = (roster.merge(pred, on='legislator_id', how='left')
                      .assign(prob_yes=lambda d: d['prob_yes'].fillna(0.5))  # neutral for missing models
                      .assign(pred_vote=lambda d: (d['prob_yes'] >= vote_threshold).astype(int))
                      .sort_values('prob_yes', ascending=False)
                      .reset_index(drop=True))

    # chamber pass estimates
    chamber_results = {}
    for chamber in chambers_to_check:
        chamber_results[chamber] = chamber_pass_estimate(pred, roster, chamber=chamber, threshold=vote_threshold)

    return {
        'per_legislator': pred_all,             # DataFrame: legislator_id, name, party, state, chamber, prob_yes, pred_vote
        'per_chamber': chamber_results,         # dict: House/Senate -> metrics + 'details' DataFrame for that chamber
        'text_used_for_inference': text,
        'threshold': vote_threshold,
    }

# --------- Optional: pretty print helpers ---------
def print_chamber_summary(ch_res: dict):
    ch = ch_res['chamber']
    members = ch_res['members']
    need = ch_res['simple_majority_needed']
    exp = ch_res['expected_yes_votes']  