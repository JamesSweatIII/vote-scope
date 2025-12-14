import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from data_sources import get_voting_data

vote_data = get_voting_data()

df = vote_data.copy()

# Standardize vote into binary (0/1/NaN)
df['vote_binary'] = df['vote'].where(df['vote'].isin([0,1]), np.nan)

# Create bill_id (Year_Roll format)
df['bill_id'] = (
    df['year'].astype(str) + "_" +
    df['roll'].astype(str).str.zfill(3)
)

# Build legislator/bill matrix
vote_matrix = df.pivot_table(
    index='legislator_id',
    columns='bill_id',
    values='vote_binary',
    aggfunc='first'
)

# Add useful columns (deduplicated)

useful_cols = [
    "legislator_id",
    "name",
    "party",
    "state"
]

party_map = (
    df[useful_cols]
    .drop_duplicates("legislator_id")
    .set_index("legislator_id")
)

vote_matrix = vote_matrix.join(party_map)

# Replace NaN (missed votes) with 0.5 ("neutral")
X = vote_matrix.drop(columns=['party', 'name', 'state']).fillna(0.5)  

# Run PCA
pca = PCA(n_components=3)
Z = pca.fit_transform(X)

# Add voter extremity column to Vote Matrix
vote_matrix['ideology_pc1'] = Z[:, 0]
vote_matrix['PC_2'] = Z[:, 1]
vote_matrix['PC_3'] = Z[:, 2]

# Z-score of ideology (mean 0, std 1)
mean_pc1 = vote_matrix['ideology_pc1'].mean()
std_pc1 = vote_matrix['ideology_pc1'].std()

mean_pc2 = vote_matrix['PC_2'].mean()
std_pc2 = vote_matrix['PC_2'].std()

mean_pc3 = vote_matrix['PC_3'].mean()
std_pc3 = vote_matrix['PC_3'].std()

vote_matrix['extremity_z'] = (vote_matrix['ideology_pc1'] - mean_pc1) / std_pc1
vote_matrix['extremity_PC2'] = (vote_matrix['PC_2'] - mean_pc2) / std_pc2
vote_matrix['extremity_PC3'] = (vote_matrix['PC_3'] - mean_pc3) / std_pc3

def plot_ideology_numberline(df, chamber="house"):

    sub = df.copy()
    sub = sub.sort_values('extremity_z')

    # Make extreme points more visible
    sub['color_value'] = sub['extremity_z']
    sub['PC2'] = sub['extremity_PC2']

    # Build figure manually for full control
    fig = go.Figure()

    # Add the points (with sharper color contrast)
    fig.add_trace(go.Scatter(
        x=sub['extremity_z'],
        y=sub['PC2'],
        mode="markers",
        marker=dict(
            size=12,
            color=sub['color_value'],
            colorscale=[
                [0.0, "#08306b"],     # dark blue
                [0.45, "#4292c6"],
                [0.50, "#f7f7f7"],    # white center
                [0.55, "#ef3b2c"],
                [1.0, "#67000d"]      # dark red
            ],
            showscale=False,
            line=dict(width=1, color="black")
        ),
        customdata=np.stack([
            sub["name"],
            sub["state"],
            sub["party"],
            sub["extremity_z"],
            sub["extremity_PC2"]
        ], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "State: %{customdata[1]}<br>" +
                      "Party: %{customdata[2]}<br>" +
                      "Party Extremity: %{customdata[3]:.3f}<br>" + 
                      "Intra-Party Factionism: %{customdata[4]:.3f}<br><extra></extra>" 
    ))

    # Zero vertical line
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=vote_matrix['extremity_PC2'].min()-0.5, y1=vote_matrix['extremity_PC2'].max()+0.5,
        line=dict(color="black", width=2, dash="dot")
    )

    # -1 vertical line
    fig.add_shape(
        type="line",
        x0=-1, x1=-1,
        y0=vote_matrix['extremity_PC2'].min()-0.5, y1=vote_matrix['extremity_PC2'].max()+0.5,
        line=dict(color="gray", width=1, dash="dot")
    )

    # +1 vertical line
    fig.add_shape(
        type="line",
        x0=1, x1=1,
        y0=vote_matrix['extremity_PC2'].min()-0.5, y1=vote_matrix['extremity_PC2'].max()+0.5,
        line=dict(color="gray", width=1, dash="dot")
    )

    # Zero horizontal line
    fig.add_shape(
        type="line",
        x0=vote_matrix['extremity_z'].min()-0.1, x1=vote_matrix['extremity_z'].max()+0.1,
        y0=0, y1=0,
        line=dict(color="black", width=2, dash="dot")
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="Principal Component Analysis - Rep. Partisanship Score",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Party Extremity (Z-Score)",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title="Intra-Party Factionism (Z-Score)",
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            range=[vote_matrix['extremity_PC2'].min()-0.5, vote_matrix['extremity_PC2'].max()+0.5]   
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor="#fafafa",   # Match .card-soft background
        paper_bgcolor="#fafafa",  # Match .card-soft background
        showlegend=False
    )

    return fig


# PC1 weights mapped to bill IDs
pc1_weights = pd.Series(pca.components_[0], index=X.columns)

# Identify strongest contributors
most_left_bill_id = pc1_weights.idxmin()
most_right_bill_id = pc1_weights.idxmax()

most_left_weight = pc1_weights.min()
most_right_weight = pc1_weights.max()

bill_titles = (
    df[['bill_id', 'title']]
    .drop_duplicates('bill_id')
    .set_index('bill_id')
)


# PC1 + PC2 weight lookup
pc1_weights = pd.Series(pca.components_[0], index=X.columns)
pc2_weights = pd.Series(pca.components_[1], index=X.columns)

# Separate Democrats from GOP
dems = vote_matrix[vote_matrix['extremity_z'] < 0]
gop = vote_matrix[vote_matrix['extremity_z'] > 0]

# Compute contributions: 
dem_pc2_contrib = (dems[X.columns] * pc2_weights).mean().sort_values(key=np.abs, ascending=False)
gop_pc2_contrib = (gop[X.columns] * pc2_weights).mean().sort_values(key=np.abs, ascending=False)

# Top 3 for each
dem_top3 = dem_pc2_contrib.index[:3]
gop_top3 = gop_pc2_contrib.index[:3]
