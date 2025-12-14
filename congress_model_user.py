import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

from data_sources import get_legislator_votes

df = get_legislator_votes()

def get_vote_matrix(df):
    vote_matrix = df.pivot_table(
        index="legislator_id",
        columns="bill_number",
        values="vote",
        aggfunc="mean"
    )
    vote_matrix = vote_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
    return vote_matrix

def get_knn_pca(df, k=5):
    vote_matrix = get_vote_matrix(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(vote_matrix)
    n_samples = X.shape[0]
    k = max(2, min(int(k), n_samples - 1))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(X)
    vote_matrix["cluster"] = clusters
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    vote_matrix["PC1"] = coords[:, 0]
    vote_matrix["PC2"] = coords[:, 1]
    return vote_matrix, scaler, pca

congress_vote_matrix, scaler, pca = get_knn_pca(df)

def plot_knn_clusters(df, k=5):
    vote_matrix, scaler, pca = get_knn_pca(df, k)
    fig = px.scatter(
        vote_matrix,
        x="PC1",
        y="PC2",
        color="cluster",
        color_discrete_sequence=[
            "#4A90E2", "#A780FF", "#D0021B", "#8E44AD", "#5DADE2", "#E74C3C"
        ],
        hover_name=vote_matrix.index,
        title=f"KMeans Voting Similarity of Congress Members (k={k})",
        height=550
    )
    fig.update_layout(
        width=None,
        height=600,
        autosize=True,
        responsive=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        title_font=dict(size=22, color="#222"),
        xaxis=dict(
            title="PC1 (Voting Behavior Dimension 1)",
            zeroline=True,
            zerolinecolor="#bbbbbb",
            zerolinewidth=2,
            showgrid=True,
            gridcolor="#e5e5e5",
            gridwidth=1,
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title="PC2 (Voting Behavior Dimension 2)",
            zeroline=True,
            zerolinecolor="#bbbbbb",
            zerolinewidth=2,
            showgrid=True,
            gridcolor="#e5e5e5",
            gridwidth=1,
            tickfont=dict(size=16),
        ),
        legend_title_text="Cluster",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig

# 1) Get the original feature columns used in PCA (bill_number columns)
n_features = pca.components_.shape[1]
bill_cols = congress_vote_matrix.columns[:n_features]
bill_cols = bill_cols.astype(int)

# 2) Build the loadings dataframe for ALL bills used in PCA
loadings = pd.DataFrame(
    pca.components_.T,              # shape: (n_features, 2)
    columns=["PC1_loading", "PC2_loading"]
)
loadings["bill_number"] = bill_cols.values

# 3) Compute influence score for each bill
loadings["influence"] = (
    np.abs(loadings["PC1_loading"]) + np.abs(loadings["PC2_loading"])
)

# 4) Build a full bill info table: loadings + metadata
bill_meta = df[["bill_number", "bill_type", "title", "summary"]].drop_duplicates()
bill_meta["bill_number"] = bill_meta["bill_number"].astype(int)

bill_info = loadings.merge(
    bill_meta,
    on="bill_number",
    how="left"
)

# 5) Start from the top 10 most influential bills
top10 = bill_info.sort_values("influence", ascending=False).head(10)

# 6) Remove specific bills you don't want
remove_bills = {7423, 2754, 3672, 8057}
remaining = top10[~top10["bill_number"].isin(remove_bills)]

# 7) Figure out how many replacements we need to get back to 10
n_to_add = 10 - len(remaining)

# 8) Sample replacement bills from the rest of bill_info
eligible = bill_info[~bill_info["bill_number"].isin(remaining["bill_number"])]
replacements = eligible.sample(n_to_add, random_state=0)

# 9) Final updated table with correct loadings for ALL 10 bills
cols = ["bill_number", "bill_type", "title", "summary",
        "PC1_loading", "PC2_loading", "influence"]

updated_top_bills_df = pd.concat(
    [remaining[cols], replacements[cols]],
    ignore_index=True
)


