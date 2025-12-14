import os
import difflib
from pathlib import Path

import pandas as pd

from data_sources import get_member_summary, get_voting_data

# ===============================
# Congress Visualization EDA Pipeline 
# ===============================

'''
This pipeline transforms raw congressional vote data into a
structured and visually explorable system for analyzing
legislative behavior and issue-based voting patterns:

1. Data Loading & Cleaning
   Import the full vote dataset, standardize member_id and bill_id,
   and normalize vote values into a consistent 0/1/NA format.

2. Core Table Construction (members_df, bills_df, votes_df)
   Split the dataset into clean relational tables:
     • members_df – unique information about each legislator
     • bills_df – metadata and text descriptions for each bill
     • votes_df – long-format table linking members to their votes

3. Roll-Call Matrix
   Convert votes_df into a member × bill voting matrix. This creates
   a structured representation of each member’s full voting pattern,
   enabling similarity analysis and clustering.

4. Voting Similarity & Clustering
   Compute cosine similarity between members’ voting vectors and apply
   KMeans clustering to identify voting blocs, ideological groupings,
   or coalitions within Congress.

5. Bill Topic Modeling (TF-IDF + SVD)
   Extract compact “topic dimensions” from bill summaries/titles by
   converting text into TF-IDF vectors, then reducing them with
   SVD. These issue dimensions allow bills (and members’ voting patterns) 
   to be compared by policy area.

6. Member-Level Issue Profiles
   Aggregate bill topic features for each member based on their voting
   behavior, producing a per-member distribution of support across
   key issue categories. These are used for visualizations such
   as radar charts.

7. Visual Outputs
   Generate intuitive graphical summaries including:
     • radar charts – visualize each member’s support across major issue areas
     • clustering maps (PCA) – place members in a 2D ideological space
     • transparency exports – save high-quality PNG/HTML outputs for dashboards
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 0. Load data
# -----------------------------------------------------------
df = get_voting_data()

# -----------------------------------------------------------
# 1. Create dataframe columns: member_id and bill_id
# -----------------------------------------------------------

'''
We generate two standardized identifier columns to ensure consistency
throughout the analysis:

• member_id: A unique, stable identifier for each legislator. Depending on the
  dataset, it may come from an existing column (e.g., legislator_id,
  bioguide_id) or be constructed from name + state as a fallback.

• bill_id: A normalized identifier for each bill. If the dataset includes a
  bill_number or legis_number column, we convert it to a string to
  avoid numeric parsing issues. If not available, we fall back to
  using the row index.
'''

if 'legislator_id' in df.columns:
    df['member_id'] = df['legislator_id'].astype(str)
elif 'member_id' in df.columns:
    df['member_id'] = df['member_id'].astype(str)
else:
    df['member_id'] = (df['name'].astype(str).fillna('') + '::' + df.get('state', '').astype(str)).astype(str)

# bill_id: prefer bill_number (string), fallback to legis_number then index

if 'bill_number' in df.columns:
    # ensure string to avoid numeric coercion errors later
    df['bill_id'] = df['bill_number'].fillna('').astype(str)
elif 'legis_number' in df.columns:
    df['bill_id'] = df['legis_number'].fillna('').astype(str)
else:
    df['bill_id'] = df.index.astype(str)

# bill_label for easy display

df['bill_label'] = df.get('legis_number', df.get('bill_number', df['bill_id'])).astype(str)


# -----------------------------------------------------------
# 2. Standardize the vote column to 'vote_clean' (0/1/NaN)
# -----------------------------------------------------------

def clean_vote(v):
    if pd.isna(v):
        return np.nan
    try:
        f = float(v)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    vstr = str(v).strip().lower()
    if vstr in {'yea','yes','aye','y'}:
        return 1
    if vstr in {'nay','no','n'}:
        return 0
    if vstr in {'present','absent','not voting','not-voting','nv'}:
        return np.nan
    return np.nan

if 'vote_clean' not in df.columns:
    if 'vote' in df.columns:
        df['vote_clean'] = df['vote'].apply(clean_vote)
    else:
        col_candidates = [c for c in df.columns if 'vote' in c.lower() or 'voted' in c.lower()]
        if col_candidates:
            df['vote_clean'] = df[col_candidates[0]].apply(clean_vote)
        else:
            raise KeyError("No vote-like column found. Provide 'vote' or similar in CSV.")


# -----------------------------------------------------------
# 3. Build dataframes: members_df, bills_df, votes_df
# -----------------------------------------------------------

''' 
These three dataframes form the core structured tables used throughout
the analysis:

• members_df: A unique list of all legislators, containing stable identifiers
  (member_id) along with attributes such as name, party, and state.
  This acts as the “lookup table” for member-level metadata.

• bills_df: A unique table of all bills appearing in the vote dataset. It includes
  identifiers (bill_id), titles, summaries, bill types, dates, and other
  metadata. Topic features (from TF-IDF + SVD) are later added here.

• votes_df: The long-format record of every individual vote. Each row represents
  one member voting on one bill. At minimum it contains member_id,
  bill_id, and the cleaned vote (0/1/NaN). This table links members_df
  and bills_df together and is the basis for building the roll-call
  matrix and computing member-level statistics.
'''

members_df = df[['member_id','name','party','state']].drop_duplicates(subset=['member_id']).reset_index(drop=True)

bill_cols = []
for c in ['bill_id','bill_label','legis_number','bill_number','bill_type','bill_type_expanded',
          'introduced_date','latest_action_date','latest_action_text','origin_chamber','title','summary','legislation_url','congress']:
    if c in df.columns:
        bill_cols.append(c)
if 'bill_id' not in bill_cols:
    bill_cols.append('bill_id')

bills_df = df[bill_cols].drop_duplicates(subset=['bill_id']).reset_index(drop=True)

votes_df = df[['member_id','bill_id','roll','year','vote_clean']].copy().rename(columns={'vote_clean':'vote'})


# -----------------------------------------------------------
# 4. Roll-call matrix (members × bills): votes_df -> roll_matrix
# -----------------------------------------------------------

'''
We reshape the raw vote data into a 2-dimensional matrix where each row
represents a member of Congress and each column represents a bill.
The cell value is that member's vote on that bill (YES/NO/NA). This matrix 
transforms individual vote records into a member-by-bill grid for the following 
voting pattern analysis.

Allows analysis of voting behavior in a structured way:
- compare members directly by their voting patterns,
- compute similarities between members,
- cluster members into voting blocs,
- and feed the data into dimensionality-reduction methods for visualization.
'''

roll_matrix = votes_df.pivot_table(index='member_id', columns='bill_id', values='vote', aggfunc='first')


# -----------------------------------------------------------
# 5. Member voting similarity & clustering: cosine similarity + KMeans
# -----------------------------------------------------------

'''
We compare members based on their full roll-call voting patterns.
First, each member is represented as a vector of votes (YES/NO/NA) across all bills.
We then compute cosine similarity between these vectors to measure how closely
two members vote overall (1.0 means identical voting patterns, 0.0 means no
alignment in voting).

After producing the member-by-member similarity matrix, we apply KMeans
clustering to group members who consistently vote in similar ways. These
clusters can reflect ideological blocs, caucus alignments, or general voting
coalitions within the chamber.'''

X_roll = roll_matrix.fillna(0).values
if X_roll.shape[0] < 2:
    sim_df = pd.DataFrame(index=roll_matrix.index, columns=roll_matrix.index, data=0.0)
else:
    sim_matrix = cosine_similarity(X_roll)
    sim_df = pd.DataFrame(sim_matrix, index=roll_matrix.index, columns=roll_matrix.index)

k = 3 if sim_df.shape[0] >= 3 else max(1, sim_df.shape[0])
if sim_df.shape[0] <= 1:
    labels = np.zeros(sim_df.shape[0], dtype=int)
else:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(sim_df)
member_clusters = pd.DataFrame({'member_id': sim_df.index, 'cluster': labels})


# -----------------------------------------------------------
# 6. Bill topic dimensions (TF-IDF + SVD on summary/title)
# -----------------------------------------------------------

'''
We convert each bill's summary/title into a numerical vector using 
TF-IDF (Term Frequency-Inverse Document Frequency), which highlights words 
that are important in that bill but not common across all bills. We also apply SVD 
(Singular Value Decomposition) to reduce them to a small number of “topic dimensions.” 
Each bill ends up with 5 numeric topic coordinates that capture its dominant themes 
(e.g., budget, defense, environment).

These topic dimensions are later averaged across the bills a member votes YES on, helping us 
quantify and visualize the policy areas each member tends to support.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
text_source = 'summary' if 'summary' in bills_df.columns and bills_df['summary'].notna().any() else 'title'

texts = bills_df[text_source].fillna('').astype(str).tolist()
if len(texts) == 0:
    for i in range(1,6):
        bills_df[f"topic_dim_{i}"] = 0.0
else:
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_mat = tfidf.fit_transform(texts)
    n_topic_dims = min(5, max(1, tfidf_mat.shape[1]-1))
    svd = TruncatedSVD(n_components=n_topic_dims, random_state=42)
    topics = svd.fit_transform(tfidf_mat)
    for i in range(topics.shape[1]):
        bills_df[f"topic_dim_{i+1}"] = topics[:, i]
    for i in range(topics.shape[1], 5):
        bills_df[f"topic_dim_{i+1}"] = 0.0


# -----------------------------------------------------------
# 7. Member & Bill stats: total votes, yes/no counts, pct yes
# -----------------------------------------------------------

member_stats = votes_df.groupby('member_id').agg(
    total_votes = ('vote','count'),
    yes_votes = ('vote', lambda s: int(np.nansum(s==1))),
    no_votes = ('vote', lambda s: int(np.nansum(s==0)))
).reset_index()
member_stats['pct_yes'] = member_stats['yes_votes'] / member_stats['total_votes'].replace({0: np.nan})
members_summary = members_df.merge(member_stats, on='member_id', how='left')

bill_stats = votes_df.groupby('bill_id').agg(
    yes_votes = ('vote', lambda s: int(np.nansum(s==1))),
    no_votes  = ('vote', lambda s: int(np.nansum(s==0))),
    turnout   = ('vote', 'count')
).reset_index()
bill_stats['pct_yes'] = bill_stats['yes_votes'] / bill_stats['turnout'].replace({0: np.nan})
bills_merged = bills_df.merge(bill_stats, on='bill_id', how='left')


# -----------------------------------------------------------
# 8. Feature engineering for modeling: merge tables
# -----------------------------------------------------------

'''
At this stage, we combine members_df, bills_df, and votes_df into a
single dataset. Each row now represents one member voting on
one bill, with both member-level attributes (party, state, cluster) and 
bill-level attributes (topic dimensions, bill type, chamber, etc.). 
By aligning all relevant features with each vote, we allow the model to
learn how characteristics of members and characteristics of bills jointly 
influence voting behavior.
'''

X = votes_df.merge(members_df, on='member_id', how='left')
X = X.merge(bills_df, on='bill_id', how='left')
X = X.merge(member_clusters, on='member_id', how='left')

# Label-encode small categorical columns (including bill_number as categorical)
label_encode_cols = ['party','state','bill_type','origin_chamber','bill_number']
for c in label_encode_cols:
    if c in X.columns:
        X[c] = X[c].fillna('UNK').astype(str)
        X[c + "_le"] = LabelEncoder().fit_transform(X[c])
# keep track of created label cols
label_cols = [c for c in X.columns if c.endswith('_le')]

# Ensure topic cols exist in X (copy over from bills_df join)
for i in range(1,6):
    col = f"topic_dim_{i}"
    if col in bills_df.columns and col not in X.columns:
        # safe merge copy
        X = X.merge(bills_df[['bill_id',col]], on='bill_id', how='left')

# Target
y = X['vote']


# -----------------------------------------------------------
# 9. DROP identifiers and non-feature columns before modeling
# -----------------------------------------------------------

'''
We remove non-predictive or identifier columns such as names, IDs, URLs, and
text fields. The goal is to keep only meaningful features such as encoded categorical 
variables, topic dimensions, and engineered voting statistics, so that the classifier 
learns patterns from member and bill characteristics rather than arbitrary identifiers.
'''

non_feature_cols = [
    'member_id','bill_id','roll','year','name','legis_number','legislator_id',
    'title','summary','latest_action_text','bill_label','legislation_url'
]
# drop any that exist
for c in non_feature_cols:
    if c in X.columns:
        X.drop(columns=c, inplace=True)

# Object columns: one-hot small-cardinality, drop large-cardinality
obj_cols = X.select_dtypes(include=['object','category']).columns.tolist()
to_one_hot = [c for c in obj_cols if X[c].nunique() <= 50]
to_drop = [c for c in obj_cols if c not in to_one_hot]

if to_one_hot:
    X = pd.get_dummies(X, columns=to_one_hot, drop_first=True)

if to_drop:
    X.drop(columns=to_drop, inplace=True)

# Fill numeric NAs (remaining numeric columns only)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_cols] = X[numeric_cols].fillna(0)

# Final feature matrix
X_feat = X[numeric_cols].copy()


# ===============================
# Visualization 
# ===============================

import os, re, math, json, textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import plotly.graph_objects as go

# ---------- configuration ----------
# OUT_DIR = "transparency_exports"
# FIG_DIR = os.path.join(OUT_DIR)
# os.makedirs(FIG_DIR, exist_ok=True)

EXPORT_PLOTLY = False   # set True to also create interactive Plotly HTML files (optional)
#TOP_N = 12             # how many top members to render (optional)
NARRATIVE_WRAP = 100    # characters per line for embedded narrative
BOX_ALPHA = 0.85        # box background alpha (0 transparent - 1 opaque)
BOX_FACECOLOR = "white" # narrative box color (white with alpha for contrast)
BOX_EDGECOLOR = "#cccccc"
TEXT_COLOR = "#222222"

# ---------- 6 unique policy area buckets ----------

'''
To organize bills into meaningful policy themes, we classify each bill
into one of six broad issue-area “buckets.” Each bucket is defined using
a small set of carefully chosen regular-expression keywords that appear
in bill titles or summaries. 

This allows us to map each bill to a consistent policy area (e.g., Economy & Budget, 
Defense & Security, Health & Public Safety, etc.). These buckets make downstream 
visualizations much more interpretable.
'''

BUCKETS = {
    "economy_budget": {
        "label": "Economy & Budget",
        "keywords": [r"\bbudget\b", r"\bappropriat", r"\btax(es|ation)?\b", r"\bfiscal\b", r"\bspending\b"]
    },
    "defense_security": {
        "label": "Defense & Security",
        "keywords": [r"\bdefense\b", r"\bsecurity\b", r"\bmilitary\b", r"\bnational security\b", r"\barmed forces\b"]
    },
    "health_safety": {
        "label": "Health & Public Safety",
        "keywords": [r"\bhealth\b", r"\bmedical\b", r"\bpublic health\b", r"\bmedicaid\b", r"\bmedicare\b", r"\bsafety\b"]
    },
    "energy_environment": {
        "label": "Energy & Environment",
        "keywords": [r"\benergy\b", r"\benvironment\b", r"\bclimate\b", r"\bepa\b", r"\bpollut"]
    },
    "procedure_ethics": {
        "label": "Procedure, Ethics & Oversight",
        "keywords": [r"\brule(s)?\b", r"\brules committee\b", r"\bresolution\b", r"\bimpeach\b",
                     r"\bcensure\b", r"\bethic(s)?\b", r"\boversight\b", r"\brules of the house\b"]
    },
    "veterans_services": {
        "label": "Veterans & Services",
        "keywords": [r"\bveteran(s)?\b", r"\bva\b", r"\bbenefit(s)?\b", r"\bservice members\b"]
    }
}
BUCKET_IDS = list(BUCKETS.keys())
BUCKET_LABELS = [BUCKETS[b]['label'] for b in BUCKET_IDS]

# ---------- map text -> bucket ----------
def match_text_to_bucket(text):
    if not text or str(text).strip() == "":
        return None
    text_l = str(text).lower()
    scores = {bid: 0 for bid in BUCKET_IDS}
    for bid, binfo in BUCKETS.items():
        for kw in binfo['keywords']:
            try:
                pat = re.compile(kw, flags=re.IGNORECASE)
                matches = pat.findall(text_l)
                if matches:
                    scores[bid] += max(1, len(matches)) * 2
            except re.error:
                if kw.lower() in text_l:
                    scores[bid] += 1
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] <= 0:
        return None
    return best[0]

# ---------- sanity check inputs ----------
required = ['bills_df', 'votes_df', 'members_df']
missing = [r for r in required if r not in globals()]
if missing:
    raise RuntimeError(f"Missing the following DataFrames: {missing}. Run the earlier pipeline first.")

# ---------- map bills -> buckets ----------
text_col = 'summary' if 'summary' in bills_df.columns and bills_df['summary'].notna().any() else 'title'
bills_df = bills_df.copy()
bills_df['_text_for_bucket'] = bills_df[text_col].fillna('').astype(str)
bills_df['_bucket_id'] = bills_df['_text_for_bucket'].apply(match_text_to_bucket).fillna('other')

# ---------- merge into votes ----------
votes_b = votes_df.merge(bills_df[['bill_id','_bucket_id']], on='bill_id', how='left')
votes_b['_bucket_id'] = votes_b['_bucket_id'].fillna('other')

# ---------- per-member per-bucket stats ----------
mb = votes_b.groupby(['member_id','_bucket_id']).agg(
    total_votes = ('vote','count'),
    yes_votes = ('vote', lambda s: int(np.nansum(s==1)))
).reset_index()
mb['pct_yes'] = np.where(mb['total_votes']>0, (mb['yes_votes']/mb['total_votes'])*100.0, np.nan)

member_bucket_pct = mb.pivot(index='member_id', columns='_bucket_id', values='pct_yes').fillna(0.0)
for bid in BUCKET_IDS:
    if bid not in member_bucket_pct.columns:
        member_bucket_pct[bid] = 0.0
member_bucket_pct = member_bucket_pct[BUCKET_IDS].reset_index()

member_counts = mb.groupby('member_id').agg(
    total_votes=('total_votes','sum'),
    yes_total=('yes_votes','sum')
).reset_index()
member_summary_df = members_df.merge(member_counts, on='member_id', how='left').merge(member_bucket_pct, on='member_id', how='left').fillna(0)

##
## ------ save CSV for dashboard---- ###
##

'''
We export the member-level summary table to a clean CSV file that provides
the main data source for interactive dashboards and visualizations.

  • Write member_summary_df to a CSV file named "member_summary_pct_by_bucket.csv"
    which contains each member’s:
      - basic info (name, party, state),
      - total YES/NO votes,
      - percent-YES values for each policy bucket.
'''

# ---------- compute global baseline (avg YES% per bucket) ----------
global_baseline = []
for bid in BUCKET_IDS:
    vals = member_summary_df[bid].replace({0:np.nan})
    avg = vals.mean()
    global_baseline.append(float(0 if np.isnan(avg) else avg))

# ----------------------
# Normalize names for matching
# ----------------------

'''
We standardize all names to a simple format to make lookup and
matching reliable. The normalizer transforms each name by:
  • converting to lowercase,
  • removing punctuation,
  • collapsing multiple spaces,
  • returning a clean “first last” style string.
'''

def normalize_name(s):
    import re
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = s.replace('.', '')  # remove periods from initials
    s = s.lower()
    s = re.sub(r"[^a-z\,\s]", "", s)      # keep letters + commas + spaces
    s = re.sub(r"\s+", " ", s).strip()    # collapse spaces
    return s

# ----------------------
# helper to normalize names to "first last" where possible
# ----------------------

def canonicalize_dataset_name(raw):
    if pd.isna(raw) or raw is None:
        return ""
    s = str(raw).strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',', 1)]
        # parts[0]=last, parts[1]=first ... -> reorder
        canon = parts[1] + " " + parts[0]
        return normalize_name(canon)
    return normalize_name(s)

# ----------------------
# name lookup + radar call
# ----------------------

def make_radar_for_full_name(
    full_name_query,
    csv_path="member_summary_pct_by_bucket.csv",
    show=True,
    save=False,
    fuzzy_suggestions=5
):
    # Load the data
    df = pd.read_csv(csv_path)
    df['name_canon'] = df['name'].fillna("").astype(str).apply(canonicalize_dataset_name)
    q_norm = normalize_name(full_name_query)
    matches = df[df['name_canon'] == q_norm]
    if matches.empty:
        return None
    if len(matches) > 1:
        cols = [c for c in ['member_id', 'name', 'party', 'state'] if c in matches.columns]
        return None

    row = matches.iloc[0]
    # Prepare radar data
    vals_yes = [float(row.get(b, 0.0)) for b in BUCKET_IDS]
    vals_yes = [0.0 if pd.isna(v) else v for v in vals_yes]
    vals_yes += vals_yes[:1]  # close the loop

    # Baseline
    base_vals = [float(global_baseline[i]) for i in range(len(BUCKET_IDS))]
    base_vals += base_vals[:1]

    # Labels
    labels = BUCKET_LABELS + [BUCKET_LABELS[0]]

    # Format party/state for subtitle
    party = row.get('party', '')
    state = row.get('state', '')
    party_state = f" ({party}–{state})" if party or state else ""

    # Plotly radar chart with improved formatting
    fig = go.Figure()

        # Member trace (make more vivid)
    fig.add_trace(go.Scatterpolar(
        r=vals_yes,
        theta=labels,
        fill='toself',
        name=row['name'],
        line=dict(color='#1976d2', width=4),           # deeper blue, thicker line
        marker=dict(size=9, color='#1976d2'),          # matching marker color
        fillcolor='rgba(25, 118, 210, 0.18)',          # blue fill, more visible
        opacity=0.95,
    ))

    # Baseline trace (make lighter and more dashed)
    fig.add_trace(go.Scatterpolar(
        r=base_vals,
        theta=labels,
        fill='toself',
        name='Global baseline',
        line=dict(color='#bbbbbb', dash='dot', width=2),  # lighter, dotted line
        marker=dict(size=8, color='#bbbbbb'),
        fillcolor='rgba(180,180,180,0.40)',               # very faint gray fill
        opacity=0.90
    ))

    fig.update_layout(
    polar=dict(
        bgcolor="#fafafa",
        domain=dict(x=[0, 1], y=[0, 1]),  # <-- maximize polar area


        radialaxis=dict(
            visible=True,
            range=[0, 100],
            showticklabels=True,
            tickfont=dict(size=12, color="#222"),   # darker ticks
            gridcolor="#b0b0b0",                    # darker gridlines
            gridwidth=1.0,
            linecolor="#888888",                    # darker axis line
            linewidth=0.6,
        ),

        angularaxis=dict(
            tickfont=dict(size=14, color="#111"),   # darker angular labels
            rotation=90,
            direction="clockwise",
            gridcolor="#b0b0b0",                    # darker gridlines
            gridwidth=1.0,
            linecolor="#888888",                    # darker axis line
            linewidth=0.6,
        ),
    ),


        # -------------------------------
        # LEGEND — moved higher for spacing
        # -------------------------------
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # move legend closer to chart
            xanchor="center",
            x=0.5,
            font=dict(size=13)
        ),

        # -------------------------------
        # TITLE — with proper spacing
        # -------------------------------
        title=dict(
            text=f"Issue Support Profile for {row['name']}{party_state}",
            font=dict(size=24, color="#222", family="system-ui"),
            x=0.5,
            xanchor="center",
            y=0.96,  # move title closer to chart
            pad=dict(t=40)       # <-- Add padding below the title
        ),

        # -------------------------------
        # MARGINS — increased top margin to prevent clipping
        # -------------------------------
        margin=dict(
            l=40,
            r=40,
            t=140,   # much less top margin
            b=80
        ),

        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",

        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
            size=13,
            color="#222"
        ),

        width=950,
        height=550,             # <-- increased for full radar visibility
        autosize=True
    )

    return fig


def prompt_for_member_and_plot(name_query=None):
    """
    Generate and display a radar chart for a member by name.
    If name_query is None, prompt the user; otherwise, use the provided name.
    """
    if not name_query:
        return
    return make_radar_for_full_name(name_query)

def _member_summary_table(csv_path="member_summary_pct_by_bucket.csv"):
    """
    Return the member summary table, preferring the precomputed dataframe or artifact.
    """
    if "member_summary_df" in globals() and member_summary_df is not None:
        return member_summary_df

    try:
        return get_member_summary()
    except Exception:
        pass

    csv_file = Path(csv_path)
    if csv_file.exists():
        return pd.read_csv(csv_file)

    raise FileNotFoundError(
        f"Member summary data not found at {csv_path} and no artifact is available. "
        "Run `python build_artifacts.py` to generate the bundle."
    )


def get_member_narrative(full_name_query, csv_path="member_summary_pct_by_bucket.csv"):
    """
    Compute narrative summary for a member (strongest/weakest buckets and deviations
    from baseline). Returns dict with lines and a combined text.
    """
    df = _member_summary_table(csv_path)
    df['name_canon'] = df['name'].fillna("").astype(str).apply(canonicalize_dataset_name)
    q_norm = normalize_name(full_name_query)
    matches = df[df['name_canon'] == q_norm]
    if matches.empty:
        return {"narrative_text": "Member not found.", "lines": []}

    row = matches.iloc[0]
    vals_yes = [float(row.get(b, 0.0)) for b in BUCKET_IDS]
    vals_yes = [0.0 if pd.isna(v) else v for v in vals_yes]
    base_vals = [float(v) for v in global_baseline]

    strongest_idx = int(np.nanargmax(vals_yes))
    weakest_idx = int(np.nanargmin(vals_yes))
    strongest_bucket = BUCKET_LABELS[strongest_idx]
    weakest_bucket = BUCKET_LABELS[weakest_idx]
    strongest_val = vals_yes[strongest_idx]
    weakest_val = vals_yes[weakest_idx]
    spread = strongest_val - weakest_val
    if spread >= 40:
        profile = "highly specialized (large spread across issue areas)"
    elif spread >= 20:
        profile = "moderately differentiated across issues"
    else:
        profile = "fairly balanced across major issue categories"

    diffs = [vals_yes[i] - base_vals[i] for i in range(len(BUCKET_IDS))]
    ranked_pos = sorted([(i, diffs[i]) for i in range(len(BUCKET_IDS))], key=lambda x: x[1], reverse=True)
    ranked_neg = sorted([(i, diffs[i]) for i in range(len(BUCKET_IDS))], key=lambda x: x[1])
    above_list = [(BUCKET_LABELS[i], diffs[i]) for i, _ in ranked_pos if diffs[i] > 0][:2]
    below_list = [(BUCKET_LABELS[i], diffs[i]) for i, _ in ranked_neg if diffs[i] < 0][:2]

    party = row.get('party', '')
    state = row.get('state', '')
    name = row.get('name', '')
    party_state = f" ({party}–{state})" if party or state else ""

    lines = [
        f"Member: {name}{party_state}",
        f"Most supportive area: {strongest_bucket} ({strongest_val:.0f}% YES).",
        f"Least supportive area: {weakest_bucket} ({weakest_val:.0f}% YES).",
        f"Overall profile: {profile}."
    ]
    if above_list:
        al = ", ".join([f"{lbl} (+{int(round(d))} pts)" for lbl, d in above_list])
        lines.append(f"Top areas above chamber avg: {al}.")
    if below_list:
        bl = ", ".join([f"{lbl} ({int(round(d))} pts)" for lbl, d in below_list])
        lines.append(f"Top areas below chamber avg: {bl}.")

    narrative_text = " ".join(lines)
    return {"narrative_text": narrative_text, "lines": lines}

def get_member_bucket_stats(full_name_query, csv_path="member_summary_pct_by_bucket.csv"):
    """
    Return per-bucket YES% for a member as a dict: {bucket_label: percent_float}.
    """
    df = _member_summary_table(csv_path)
    df['name_canon'] = df['name'].fillna("").astype(str).apply(canonicalize_dataset_name)
    q_norm = normalize_name(full_name_query)
    matches = df[df['name_canon'] == q_norm]
    if matches.empty:
        return {}

    row = matches.iloc[0]
    stats = {}
    for i, bid in enumerate(BUCKET_IDS):
        lbl = BUCKET_LABELS[i]
        val = row.get(bid, 0.0)
        try:
            stats[lbl] = float(0.0 if pd.isna(val) else val)
        except Exception:
            stats[lbl] = 0.0
    return stats
