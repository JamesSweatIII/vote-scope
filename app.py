"""
Legislative Vote Prediction Explorer
------------------------------------

Shiny for Python application for interactively exploring predicted
legislator votes on proposed bills using pre-trained models.

Run locally (from project root):
    shiny run --reload app.py
    # or:
    python app.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from data_sources import get_voting_data
from predict_new_bill import evaluate_new_bill
from partisanship_model import plot_ideology_numberline, vote_matrix
from attribute_map import prompt_for_member_and_plot, get_member_narrative, get_member_bucket_stats
import congress_model_user
from congress_model_user import get_knn_pca, updated_top_bills_df
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Names to exclude from chamber-level voting estimates
EXCLUDED_LEGISLATORS = {
    "Hernandez",
    "King-Hinds",
    "Moylan",
    "Norton",
    "Plaskett",
    "Radewagen",
    "Fine",
    "Patronis",
}

# Pre-curated bills that the user can quickly select
BILLS: Dict[str, Dict[str, str]] = {
    "wildfire": {
        "label": "Wildfire Mitigation and Resilience Act",
        "summary": """
This bill authorizes emergency appropriations for wildfire mitigation, expands federal support
for forest management, and establishes new grants for prescribed burns and community resilience.
""",
        "title": "Wildfire Mitigation and Resilience Act",
        "type": "House Bill",
    },
    "heartbeat": {
        "label": "Heartbeat Protection Act",
        "summary": """
This bill imposes federal restrictions on abortion procedures and 
prohibits the performance of an abortion after a fetal heartbeat 
is detected, except in cases where the life of the pregnant woman
is at risk. The bill establishes criminal penalties for physicians
who knowingly perform an abortion in violation of these restrictions.
Additionally, the bill allows civil actions against providers by certain
family members and authorizes states to enforce the provisions through 
private rights of action. The Department of Health and Human Services 
must issue guidance to states regarding enforcement and reporting requirements.
""",
        "title": "Heartbeat Protection Act",
        "type": "House Bill",
    },
    "immigration": {
        "label": "Border Security and Immigration Reform Act",
        "summary": """
This bill enhances border security by increasing personnel and technology
at ports of entry, revises asylum procedures, and creates a pathway to legal
status for certain undocumented immigrants brought to the United States
as children. The bill also authorizes grants to states and localities to 
support integration and workforce development programs.
""",
        "title": "Border Security and Immigration Reform Act",
        "type": "House Bill",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

try:
    voting_df_full: pd.DataFrame = get_voting_data()
    DATA_LOAD_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001
    voting_df_full = pd.DataFrame()
    DATA_LOAD_ERROR = (
        "Error loading voting data. "
        "Run `python build_artifacts.py` or ensure the artifact/CSV is present. "
        f"Details: {exc}"
    )


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def majority_needed(n_members: int) -> int:
    """
    Simple majority: floor(n/2) + 1.
    """
    return (n_members // 2) + 1


def chamber_pass_estimate(
    pred_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    chamber: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Compute pass/fail estimates for a given chamber using thresholded probabilities.

    Parameters
    ----------
    pred_df:
        DataFrame with at least columns: ["legislator_id", "prob_yes"].
    roster_df:
        DataFrame with legislator metadata (must include "legislator_id", "name",
        "chamber", and ideally "party" and "state").
    chamber:
        Chamber name to filter on (e.g. "House", "Senate").
    threshold:
        Probability threshold at or above which a legislator is counted as a YES vote.
    """
    chamber_df = roster_df.copy()

    # Remove delegates / non-voting members or excluded names.
    chamber_df = chamber_df[~chamber_df["name"].isin(EXCLUDED_LEGISLATORS)]

    # Filter for selected chamber (case-insensitive).
    chamber_df = chamber_df[chamber_df["chamber"].str.lower() == chamber.lower()]

    # Attach predicted probabilities.
    chamber_df = chamber_df.merge(pred_df, on="legislator_id", how="left")

    # For legislators without a model, assume neutral 0.5 probability.
    chamber_df["prob_yes"] = chamber_df["prob_yes"].fillna(0.5)

    # Convert probabilities to thresholded yes/no votes.
    chamber_df["pred_vote"] = (chamber_df["prob_yes"] >= threshold).astype(int)

    n_members = len(chamber_df)
    majority = majority_needed(n_members)
    expected_yes = int(chamber_df["pred_vote"].sum())
    would_pass = expected_yes >= majority

    return {
        "chamber": chamber,
        "members": n_members,
        "simple_majority_needed": majority,
        "expected_yes_votes": expected_yes,
        "would_pass_simple_majority": bool(would_pass),
        "details": (
            chamber_df.sort_values("prob_yes", ascending=False)
            .reset_index(drop=True)
        ),
    }


def get_nearest_legislators(vote_matrix, user_PC1, user_PC2, cluster_name_map, df):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    # Ensure the 'knn_cluster' column exists
    if "knn_cluster" not in vote_matrix.columns:
        kmeans = KMeans(n_clusters=5, random_state=0)
        vote_matrix["knn_cluster"] = kmeans.fit_predict(vote_matrix[["PC1", "PC2"]])

    # Compute distance from each legislator to the user in PC1–PC2 space
    user_point = np.array([user_PC1, user_PC2])

    dist_df = vote_matrix[["PC1", "PC2", "knn_cluster"]].copy()
    dist_df["distance_to_user"] = np.sqrt(
        (dist_df["PC1"] - user_PC1) ** 2 +
        (dist_df["PC2"] - user_PC2) ** 2
    )

    # Take the 5 closest legislators
    nearest = dist_df.nsmallest(5, "distance_to_user")

    # Attach readable cluster labels (using the same map as the plot)
    nearest["cluster_label"] = nearest["knn_cluster"].map(cluster_name_map)

    # Bring in legislator names and parties
    leg_meta = df[["legislator_id", "name", "party"]].drop_duplicates("legislator_id")

    nearest = nearest.merge(
        leg_meta,
        left_index=True,          # vote_matrix index is legislator_id
        right_on="legislator_id",
        how="left"
    )

    # Show nearest neighbors with key info
    cols_to_show = ["legislator_id", "name", "party",
                    "cluster_label", "distance_to_user"]

    return nearest[cols_to_show]

# ---------------------------------------------------------------------------
# UI definition (styled like your Next.js page)
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    # ---------- Inline CSS for the sleek card layout ----------
    ui.tags.style(
        """
        :root {
          --bg-neutral-50: #fafafa;
          --bg-neutral-100: #f5f5f5;
          --bg-neutral-200: #e5e5e5;
          --bg-neutral-300: #d4d4d4;
          --border-neutral-300: #d4d4d4;
          --text-neutral-800: #262626;
          --text-neutral-600: #525252;
          --text-neutral-500: #737373;
        }

        body {
          background: radial-gradient(circle at top, #e5e5e5, #f5f5f5);
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
          color: var(--text-neutral-800);
        }

        .app-shell {
          min-height: 100vh;
          display: flex;
          align-items: stretch;
          justify-content: center;
          padding: 1.2rem 0.5rem; /* Decreased vertical and horizontal padding */
        }

        .app-container {
          position: relative;
          z-index: 10;
          width: 100%;
          max-width: 1500px; /* Wider container */
          min-height: 80vh;
          display: flex;
          flex-direction: column;
          border-radius: 1.75rem;
          border: 1px solid var(--border-neutral-300);
          background: linear-gradient(135deg, #f5f5f5, #e7e7e7);
          box-shadow:
            0 24px 80px rgba(0,0,0,0.12),
            0 0 0 1px rgba(255,255,255,0.8);
          padding: 1rem 1rem 1.5rem; /* Decreased padding */
        }
            
        .app-header {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
        }

        .app-tag {
          display: inline-flex;
          align-items: center;
          border-radius: 9999px;
          border: 1px solid var(--border-neutral-300);
          padding: 0.25rem 0.75rem;
          font-size: 0.65rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-neutral-500);
        }

        .app-title {
          font-size: 1.9rem;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          line-height: 1.1;
          color: var(--text-neutral-800);
        }

        .app-subtitle {
          max-width: 34rem;
          font-size: 0.85rem;
          color: var(--text-neutral-600);
        }

        .app-main-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.8fr) minmax(0, 1fr);
          gap: 1.75rem;
          flex: 1;
          align-items: stretch;
          margin-top: 0.5rem;
        }
        .app-main-grid > div {
          height: 100%; 
        }
        .card-soft {
          border-radius: 1.5rem;
          border: 1px solid var(--border-neutral-300);
          background: radial-gradient(circle at top, #f1f1f1, #fafafa);
          padding: 1.4rem 1.35rem;
          height: 100%; /* Make card fill parent column */
          display: flex;
          flex-direction: column;
        }

        .legend-label {
          font-size: 0.65rem;
          letter-spacing: 0.2em;
          text-transform: uppercase;
          color: var(--text-neutral-500);
          margin-bottom: 0.35rem;
        }

        .pill-button-primary {
          border-radius: 9999px;
          border: 1px solid #404040;
          background-color: rgba(0,0,0,0.7);
          padding: 0.45rem 1.2rem;
          font-size: 0.65rem;
          letter-spacing: 0.2em;
          text-transform: uppercase;
          color: #f5f5f5;
          cursor: pointer;
          transition: background-color 0.15s ease, transform 0.08s ease;
        }

        .pill-button-primary:hover {
          background-color: rgba(0,0,0,0.9);
          transform: translateY(-1px);
        }

        .meta-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
          margin-top: 1rem;
          font-size: 0.65rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-neutral-500);
        }

        .meta-pill {
          padding: 0.25rem 0.7rem;
          border-radius: 9999px;
          border: 1px dashed var(--border-neutral-300);
          background: rgba(245,245,245,0.9);
        }

        .results-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.9rem;
        }

        .results-chip {
          border-radius: 9999px;
          background-color: var(--bg-neutral-100);
          padding: 0.25rem 0.8rem;
          font-size: 0.65rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-neutral-600);
        }

        /* Style the verbatim text blocks by ID */
        #bill_text_display {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 0.78rem;
          background-color: #f4f4f4;
          border-radius: 0.9rem;
          padding: 0.9rem 0.9rem;
          border: 1px solid #e0e0e0;
          max-height: 300px;
          overflow: auto;
          white-space: pre-wrap;
        }

        #chamber_summary {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
          font-size: 0.78rem;
          background-color: #f4f4f4;
          border-radius: 0.9rem;
          padding: 0.9rem 0.9rem;
          border: 1px solid #e0e0e0;
          max-height: 130px;
          overflow: auto;
          white-space: pre-wrap;
        }

        .custom-bill-shell {
          margin-top: 2rem;
        }

        .custom-bill-card {
          border-radius: 1.5rem;
          border: 1px solid var(--border-neutral-300);
          background: var(--bg-neutral-50);
          padding: 1.2rem 1.4rem;
          font-size: 0.8rem;
          color: var(--text-neutral-600);
        }
        
        /* Force Plotly widget to full width */
        #kmeans_plot, 
        #kmeans_plot > div, 
        #kmeans_plot .js-plotly-plot {
            width: 100% !important;
        }

        ul li:hover {
            background-color: #e0e0e0; /* Light gray background on hover */
        }

        """
    ),

    ui.div(
        {"class": "app-shell"},
        ui.div(
            {"class": "app-container"},
            # HEADER
            ui.div(
                {"class": "app-header", "style": "align-items: center;"},
                ui.p("Congress · Vote Modeling · Explorer", class_="app-tag"),
                ui.h2("Legislative Vote Prediction Explorer", class_="app-title"),
                ui.p(
                    "Interactively stress-test how members of Congress are predicted to "
                    "vote on different policy proposals. Adjust thresholds, swap bills, "
                    "and compare expected outcomes by chamber.",
                    class_="app-subtitle",
                ),
            ),

            # MAIN GRID (left: controls; right: results; middle: new box)
            ui.div(
                {
                    "class": "app-main-grid",
                    "style": "display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 1.75rem; flex: 1; align-items: stretch; margin-top: 0.5rem; margin-bottom:2.5rem;"

                },
                # ===================== LEFT COLUMN: CONFIG / CONTROLS =====================
                ui.div(
                    {
                        "class": "card-soft",
                        "style": "padding:1.75rem; display:flex; flex-direction:column; gap:1.4rem;",
                    },
                    # ---- Header ----
                    ui.tags.div(
                        ui.p("CONFIGURATION", class_="legend-label"),
                    ),
                    # ---- BILL SELECTION ----
                    ui.tags.div(
                        ui.p("Bill Selection", style="font-size:0.9rem; margin-bottom:0.75rem;"),
                        ui.input_radio_buttons(
                            "bill_choice",
                            None,
                            choices={
                                "wildfire": BILLS["wildfire"]["label"],
                                "heartbeat": BILLS["heartbeat"]["label"],
                                "immigration": BILLS["immigration"]["label"],
                                "custom": "Custom bill (enter your own text)",
                            },
                            selected="heartbeat",
                            width="100%",  # <-- FULL WIDTH
                        ),
                    ),
                    # ---- DECISION THRESHOLD ----
                    ui.tags.div(
                        ui.p("Decision Threshold", style="font-size:0.9rem; margin-bottom:0.5rem;"),
                        ui.input_slider(
                            "threshold",
                            "Vote threshold (P(YES) ≥ threshold → YES)",
                            min=0.10,
                            max=0.90,
                            value=0.50,
                            step=0.05,
                            width="100%",  # <-- FULL WIDTH
                        ),
                    ),
                    # ---- CHAMBERS ----
                    ui.tags.div(
                        ui.p("Chambers", style="font-size:0.9rem; margin-bottom:0.75rem;"),
                        ui.input_checkbox_group(
                            "chambers",
                            None,
                            choices=["House", "Senate"],
                            selected=["House"],
                            width="100%",  # <-- FULL WIDTH
                        ),
                    ),
                    # ---- CONGRESS NUMBER ----
                    ui.tags.div(
                        ui.p("Congress Number", style="font-size:0.9rem; margin-bottom:0.75rem;"),
                        ui.input_numeric(
                            "congress",
                            None,
                            value=119,
                            min=100,
                            max=200,
                            width="100%",  # <-- FULL WIDTH
                        ),
                    ),
                ),
                # ===================== RIGHT COLUMN (Custom Bill) =====================
                ui.div(
                    {
                        "class": "card-soft",
                        "style": "padding:1.75rem; display:flex; flex-direction:column; gap:1.4rem;",
                    },
                    # Header
                    ui.tags.div(
                        ui.p("CUSTOM BILL SCENARIO", class_="legend-label"),
                    ),
                    ui.p(
                        "Create your own bill by entering a title and summary below. "
                        "Select 'Custom bill' in the configuration panel and click Evaluate.",
                        style="font-size:0.88rem; color:#555; line-height:1.45; margin-bottom:0.2rem;",
                    ),
                    # FULL-WIDTH INPUT BOXES
                    ui.input_text(
                        "bill_title",
                        "Bill Title",
                        placeholder="Enter the bill title...",
                        width="100%",  # stretch full width
                    ),
                    ui.input_text_area(
                        "bill_summary",
                        "Bill Summary",
                        placeholder="Summarize the bill here...",
                        width="100%",
                        rows=12,
                    ),
                ),
            ),
            # ---------- RESULTS ----------
                ui.div(
                    ui.div(
                        {"class": "card-soft"},
                        ui.div(
                            {"class": "results-card-header"},
                            ui.tags.div(
                                ui.p("Bill overview", class_="legend-label"),
                            ),
                            ui.span("Live model outputs", class_="results-chip"),
                        ),
                        ui.output_text_verbatim("bill_text_display"),
                        ui.br(),
                        ui.div(
                            ui.p("Chamber-level summary", class_="legend-label"),
                        ),
                        ui.output_text_verbatim("chamber_summary"),
                        # --- Per-Legislator Predictions Section ---
                        ui.div(
                            {
                                "style": (
                                    "margin-top:1.5rem; "
                                    "display:flex; "
                                    "flex-direction:row; "
                                    "justify-content:space-between; "
                                    "align-items:flex-end; "
                                    "gap:1rem; "
                                    "width:100%; "
                                )
                            },

                        # LEFT SIDE — Title + Chamber Select + Table (stacked vertically)
                        ui.div(
                            {
                                "style": (
                                    "display:flex; "
                                    "flex-direction:column; "
                                    "flex:1; "
                                    "gap:0.6rem;"
                                )
                            },
                            ui.p("Per-legislator predictions", class_="legend-label"),

                            ui.input_select(
                                "detail_chamber",
                                "Chamber to show details for",
                                choices=["House", "Senate"],
                                selected="House",
                            ),

                            ui.output_ui("legislator_table_text"),
                        ),

                        # RIGHT SIDE — Evaluate Button (fixed width)
                        ui.div(
                            ui.input_action_button(
                                "run_btn",
                                "Evaluate Bill",
                                class_="pill-button-primary",
                                style="width:220px; padding:0.6rem 1rem;"
                            ),
                            style="flex:none ;"
                        ),
                    ),

                    )
                ),
            # ---------- IDEOLOGY PLOT SECTION ----------
            ui.div(
    {
        "class": "card-soft",
        "style": (
            "margin-top:2rem;"
            "width:100%; max-width:1800px; margin-left:auto; margin-right:auto; padding:2.5rem 2rem;"
        ),
    },

            # ----- Section Title -----
            ui.tags.div(
                ui.p(
                    "Ideology number line (partisanship model)",
                    class_="legend-label",
                    style="margin-bottom: 1rem;",
                ),
            ),

            # ----- The Plot -----
            output_widget("ideology_plot"),

            # ----- Highlight Controls -----
            ui.div(
                {
                    "style": (
                        "margin-top:1.5rem; "
                        "display:flex; "
                        "flex-direction:row; "
                        "justify-content:space-between; "
                        "align-items:center; "   # <- ensures perfect parallel alignment
                        "gap:1rem; "
                        "width:100%; "
                    )
                },

                # Text input (fills remaining space)
                ui.div(
                    {
                        "style": "position: relative; width: 100%;",
                    },
                    ui.input_text(
                        "highlight_name_input",  # Updated ID
                        "Highlight member on graph:",
                        placeholder="e.g., Cline, Pelosi, or Johnson",
                        width="100%",
                    ),
                    ui.div(
                        ui.output_ui("highlight_name_suggestions"),
                        style=(
                            "position: absolute; top: 100%; left: 0; right: 0; "
                            "background: white; border: 1px solid #ccc; z-index: 1000; "
                            "max-height: 200px; overflow-y: auto;"
                        ),
                    ),
                ),

                # Button (fixed width, same baseline)
                ui.div(
                    ui.input_action_button(
                        "highlight_btn",
                        "Highlight",
                        class_="pill-button-primary",
                        style="width:220px; padding:0.6rem 1rem;",
                        width="10%"
                    ),
                    style="flex:none;"  # <- button stays compact
                ),
            ),
        ),

            # ---------- MEMBER ATTRIBUTE EXPLORER SECTION ----------
            ui.div(
                {
                    "style": (
                        "display:grid; "
                        "grid-template-columns: minmax(0, 3fr) minmax(0, 1fr); "  # 75% / 25%
                        "gap:2rem; "
                        "width:100%; "
                        "margin-bottom:2rem;"
                        
                    )
                },
                # Left: 75% Member Attribute Explorer
                ui.div(
                    {
                        "class": "card-soft",
                        "style": (
                            "margin-top:2rem; "
                            "width:100%; "
                            "padding:2.5rem 2rem; "
                            "min-height:600px; "
                        ),
                    },
                    ui.tags.div(
                        ui.p("Member Attribute Explorer", class_="legend-label", style="margin-bottom: 1rem;"),
                    ),
                    ui.div(
                        {
                            "style": (
                                "display:flex; "
                                "justify-content:space-between; "
                                "align-items:flex-end; "
                                "gap:1rem; "
                                "width:100%; "
                                "margin-bottom:1rem;"
                            )
                        },
                        ui.div(
                            {
                                "style": "position: relative; width: 100%;",
                            },
                            ui.input_text(
                                "member_query_input",
                                "Enter Member Name:",
                                placeholder="e.g., Collins",
                                width="100%",
                            ),
                            ui.div(
                                ui.output_ui("member_query_suggestions"),
                                style=(
                                    "position: absolute; top: 100%; left: 0; right: 0; "
                                    "background: white; border: 1px solid #ccc; z-index: 1000; "
                                    "max-height: 200px; overflow-y: auto;"
                                ),
                            ),
                        ),
                        ui.div(
                            ui.input_action_button(
                                "member_go",
                                "Generate Member Plot",
                                class_="pill-button-primary",
                                style="width:220px; padding:0.6rem 1rem;"
                            ),
                            style="flex:none;"
                        )
                    ),
                    ui.div(
                        output_widget("member_plot"),
                        style=(
                            "margin-top:1rem; "
                            "width:100%; "
                            "height:100%; "
                            "display:flex; "
                            "justify-content:center; "
                        )
                    ),
                ),
                # Right: 25% Narrative card
                ui.div(
                    {
                        "class": "card-soft",
                        "style": (
                            "margin-top:2rem; "
                            "padding:2.5rem 2rem; "
                            "width:100%; "
                        ),
                    },
                    ui.tags.div(
                        ui.p("Member Narrative", class_="legend-label", style="margin-bottom: 1rem;"),
                    ),
                    ui.output_ui("member_narrative_card")
                ),
            ),

            # K Means Clustering Section (full width)
            ui.div(
                        {"class": "card-soft", "style": "margin-top:2rem; padding:2rem;"},
                        ui.tags.div(
                            ui.p("K-Means Voting Clusters", class_="legend-label", style="margin-bottom:1rem;")
                        ),
                        ui.div(
                {
                    "style": (
                        "display:flex; "
                        "flex-direction:row; "
                        "justify-content:space-between; "
                        "align-items:flex-end; "   # <-- aligns both on same baseline
                        "width:100%; "
                        "gap:1rem; "
                        "margin-bottom:1rem;"
                    )
                },

                # # LEFT — numeric input
                # ui.div(
                #     ui.input_numeric(
                #         "k_clusters",
                #         "Number of clusters (k)",
                #         value=5,
                #         min=2,
                #         max=10,
                #     ),
                #     style="flex:1;"   # <-- allows input to take natural width
                # ),

                # # RIGHT — button
                # ui.div(
                #     ui.input_action_button(
                #         "kmeans_go",
                #         "Recalculate Clusters",
                #         class_="pill-button-primary",
                #     ),
                #     style="flex:none;"  # <-- button stays fixed size
                # ),
            ),

                ui.div(
                  
                    output_widget("kmeans_plot"),
                    style=(
                        "margin-top:1rem; "
                        "width:100%; "
                        "display:block; "      # IMPORTANT: prevents auto-shrink!
                        "overflow-x:auto; "      # <-- prevents squeezing
                        "text-align:center; "  # centers the chart
                    )
                )
            ),
            
            # 2 50/50 app cards side by side
            ui.div(
                {
                    "style": "display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr);; gap: 1.75rem; flex: 1; align-items: stretch; margin-top: 0.5rem; margin-bottom:1.5rem;"

                    
                },
                # First 35% card
                ui.div(
                    {
                        "class": "card-soft",
                        "style": (
                            "margin-top:2rem; "
                            "padding:2.5rem 2rem; "
                            "max-height:600px; "
                        ),
                    },
                    ui.tags.div(
                        ui.p("Top 10 Most Influential Bills (PCA)", class_="legend-label", style="margin-bottom: 1rem;"),
                    ),
                    ui.output_ui("top10_bills_table"),
                ),
                # Last 50% card
                ui.div(
                    {
                        "class": "card-soft",
                        "style": (
                            "margin-top:2rem; "
                            "padding:2.5rem 2rem; "
                            "max-height:600px;"
                        ),
                    },
                    ui.tags.div(
                        ui.p("Your Votes on Top 10 Influential Bills", class_="legend-label", style="margin-bottom: 1rem;"),
                    ),
                    ui.output_ui("user_bill_vote_table"),
                ),
                  # Middle 15% card
                ui.div(
                    {
                        "class": "card-soft",
                        "style": (
                            "margin-top:2rem; "
                            "padding:2.5rem 2rem; "
                            "max-height:600px; "
                        ),
                    },
                    ui.tags.div(
                        ui.p("Top 5 Most Similar Members", class_="legend-label", style="margin-bottom: 1rem;"),
                    ),
                    ui.output_ui("similar_legislators"),
                )
            ),
          
        ),
    ),
    ui.tags.script(
        """
        document.addEventListener('DOMContentLoaded', function() {
            // Prevent default scrolling behavior for the "Next" button
            document.getElementById('next_vote_btn').addEventListener('click', function(event) {
                event.preventDefault(); // Prevent the default action
            });
        });
        """
    )
)


# ---------------------------------------------------------------------------
# Server logic
# ---------------------------------------------------------------------------

def server(input, output, session) -> None:
    """
    Shiny server function that wires inputs → reactive calculations → outputs.
    """
    # Reactive store for evaluation results
    results_store: reactive.Value = reactive.Value(None)

    # Reactive value to store filtered suggestions
    filtered_names = reactive.Value([])

    # Separate reactive values for each input box
    highlight_suggestions = reactive.Value([])
    member_suggestions = reactive.Value([])

    # Reactive value to store the clicked name from the dropdowns
    clicked_highlight_name = reactive.Value(None)
    clicked_member_name = reactive.Value(None)

    # Clear the previously clicked name when the user starts typing again (to avoid using partials)
    @reactive.effect
    @reactive.event(input.member_query_input)
    def _clear_clicked_member_on_typing():
        # If the user types, require a new click before processing
        clicked_member_name.set(None)

    # Capture EXACT clicked member suggestion
    @reactive.effect
    @reactive.event(input.clicked_member_name)
    def _capture_clicked_member():
        name = input.clicked_member_name() or None
        clicked_member_name.set(name)

    # -----------------------------
    # Current bill configuration
    # -----------------------------
    @reactive.calc
    def current_bill() -> Dict[str, str]:
        """Return the currently selected bill (pre-defined or custom)."""
        choice = input.bill_choice()

        if choice == "custom":
            # Use the custom inputs for title and summary
            summary = (input.bill_summary() or "").strip()
            title = (input.bill_title() or "Custom Bill").strip()
            bill_type = "Custom Bill"  # Default type for custom bills
        else:
            # Use the pre-defined bill configuration
            cfg = BILLS[choice]
            summary = cfg["summary"].strip()
            title = cfg["title"]
            bill_type = cfg["type"]

        return {
            "summary": summary,
            "title": title,
            "bill_type": bill_type,
        }

    # -----------------------------
    # Run evaluation when button is clicked
    # -----------------------------
    @reactive.Effect
    @reactive.event(input.run_btn)
    def _run_evaluation() -> None:
        # If data failed to load, do not run models.
        if DATA_LOAD_ERROR is not None:
            results_store.set({"error": DATA_LOAD_ERROR})
            return

        bill = current_bill()  # Use the dynamically updated bill
        threshold: float = float(input.threshold())
        chambers: List[str] = list(input.chambers()) or ["House"]
        congress: int = int(input.congress())

        # Call existing evaluation function from predict_new_bill
        res = evaluate_new_bill(
            summary=bill["summary"],
            voting_df=voting_df_full,
            title=bill["title"],
            bill_type_expanded=bill["bill_type"],
            models_path_prefix="legislator_vote_models",
            vote_threshold=threshold,
            chambers_to_check=chambers,
            congress=congress,
        )

        # Enrich with chamber-level pass/fail estimates
        if "per_legislator" in res:
            per_leg = res["per_legislator"].copy()

            # Roster metadata without model-specific columns
            roster_df = per_leg.drop(
                columns=["prob_yes", "pred_vote"], errors="ignore"
            )

            # Prediction-only frame
            pred_df = per_leg[["legislator_id", "prob_yes"]].copy()

            per_chamber: Dict[str, Dict[str, object]] = {}
            for ch in chambers:
                per_chamber[ch] = chamber_pass_estimate(
                    pred_df=pred_df,
                    roster_df=roster_df,
                    chamber=ch,
                    threshold=threshold,
                )

            res["per_chamber"] = per_chamber

        # Add threshold into the result explicitly for display
        res["threshold"] = threshold
        results_store.set(res)

    # -----------------------------
    # Outputs
    # -----------------------------
    @output
    @render.text
    def bill_text_display() -> str:
        """
        Text block describing the currently selected bill:
        title, type, and summary.
        """
        bill = current_bill()
        lines = [
            f"Title: {bill['title']}",
            f"Type:  {bill['bill_type']}",
            "",
            "Summary:",
            bill["summary"],
        ]
        return "\n".join(lines)

    @output
    @render.text
    def chamber_summary() -> str:
        """
        High-level summary of pass/fail expectations by chamber.
        """
        res = results_store.get()

        if res is None:
            if DATA_LOAD_ERROR is not None:
                return f"Data error: {DATA_LOAD_ERROR}"
            return "No results yet. Choose a bill and click 'Evaluate bill'."

        if "error" in res:
            return f"Error: {res['error']}"

        per_chamber = res.get("per_chamber")
        if not per_chamber:
            return "No chamber-level results available. Run an evaluation first."

        threshold = res.get("threshold", 0.5)
        lines = [f"Decision threshold: P(YES) ≥ {threshold:.2f}", ""]

        for ch_name, ch_res in per_chamber.items():
            members = ch_res["members"]
            need = ch_res["simple_majority_needed"]
            exp = ch_res["expected_yes_votes"]
            passed = ch_res["would_pass_simple_majority"]
            status = "PASS" if passed else "FAIL"

            lines.append(
                f"{ch_name}: members={members}, "
                f"needed_for_simple_majority={need}, "
                f"expected_yes={exp}, "
                f"outcome={status}"
            )

        return "\n".join(lines)

    @output
    @render.table
    def legislator_table() -> pd.DataFrame:
        """
        Per-legislator prediction table for the selected chamber.
        """
        res = results_store.get()
        if res is None or "per_legislator" not in res:
            return pd.DataFrame({"info": ["No results yet."]})

        detail_chamber = input.detail_chamber()

        if "per_chamber" in res and detail_chamber in res["per_chamber"]:
            df = res["per_chamber"][detail_chamber]["details"].copy()
        else:
            df = res["per_legislator"].copy()
            df = df[df["chamber"].str.lower() == detail_chamber.lower()]

        # Columns to display in the table
        desired_cols = [
            "legislator_id",
            "name",
            "party",
            "state",
            "chamber",
            "prob_yes",
            "pred_vote",
        ]
        available_cols = [c for c in desired_cols if c in df.columns]
        df = df[available_cols].copy()

        if "prob_yes" in df.columns:
            df = df.sort_values("prob_yes", ascending=False)

        return df.reset_index(drop=True).head(50)

    @output
    @render.ui
    def legislator_table_text():
        res = results_store.get()
        if res is None or "per_legislator" not in res:
            return ui.HTML("<p>No results yet. Choose a bill and click 'Evaluate bill'.</p>")

        detail_chamber = input.detail_chamber()

        if "per_chamber" in res and detail_chamber in res["per_chamber"]:
            df = res["per_chamber"][detail_chamber]["details"].copy()
        else:
            df = res["per_legislator"].copy()
            df = df[df["chamber"].str.lower() == detail_chamber.lower()]

        cols = ["name", "party", "state", "chamber", "prob_yes", "pred_vote"]
        df = df[cols].copy()
        df["prob_yes"] = df["prob_yes"].map(lambda x: f"{x:.3f}")

        # ---- Build HTML Table ----
        html = """
        <style>
            .pro-table-container {
                max-height: 420px; 
                overflow-y: auto;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                margin-top: 0;
            }

            .pro-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }

            .pro-table thead {
                background: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 2;
            }

            .pro-table th, .pro-table td {
                padding: 10px 14px;
                border-bottom: 1px solid #e5e5e5;
            }

            .pro-table tbody tr:nth-child(even) {
                background: #fafafa;
            }

            .pro-table tbody tr:hover {
                background: #eaeaea;
            }

            .pro-table td.num {
                text-align: right;
                font-family: monospace;
            }

            .pro-table td.vote {
                text-align: center;
                font-weight: 600;
            }

            .pro-table td.vote.yes {
                color: #006400;
            }
            .pro-table td.vote.no {
                color: #8b0000;
            }
        </style>

        <div class="pro-table-container">
        <table class="pro-table">
            <thead>
                <tr>
        """

        # Add headers
        for c in cols:
            html += f"<th>{c}</th>"
        html += "</tr></thead><tbody>"

        # Add rows
        for _, row in df.iterrows():
            yes_no_class = "yes" if str(row["pred_vote"]) == "1" else "no"
            html += "<tr>"
            html += f"<td>{row['name']}</td>"
            html += f"<td>{row['party']}</td>"
            html += f"<td>{row['state']}</td>"
            html += f"<td>{row['chamber']}</td>"
            html += f"<td class='num'>{row['prob_yes']}</td>"
            html += f"<td class='vote {yes_no_class}'>{row['pred_vote']}</td>"
            html += "</tr>"

        html += "</tbody></table></div>"

        return ui.HTML(html)

    # Store the currently requested highlighted member (string or None)
    highlighted_member = reactive.Value(None)

    # Keep an internal reactive value that mirrors the last clicked suggestion
    clicked_highlight_name = reactive.Value(None)

    # When a suggestion is clicked, update our internal store with EXACT name clicked.
    @reactive.effect
    @reactive.event(input.clicked_highlight_name)
    def _capture_clicked_highlight():
        name = input.clicked_highlight_name() or None
        clicked_highlight_name.set(name)

    # Only update the graph when the Highlight button is pressed,
    # using the last clicked suggestion (not what’s typed).
    @reactive.effect
    @reactive.event(input.highlight_btn)
    def _apply_highlight_selection():
        name = clicked_highlight_name.get()
        if name:
            highlighted_member.set(name)
        else:
            highlighted_member.set(None)

    @reactive.effect
    @reactive.event(input.highlight_btn)
    def _update_highlighted_member():
        # Get the value from the clicked name
        name = clicked_highlight_name.get()
        if name:
            highlighted_member.set(name)
        else:
            highlighted_member.set(None)

    # ----------------------------------------------------
    # CLEAR RESULTS WHEN BILL CHOICE CHANGES
    # ----------------------------------------------------
    @reactive.Effect
    @reactive.event(input.bill_choice)
    def _reset_results_on_bill_change():
        results_store.set(None)   # Clear all results


    @output
    @render_widget
    def ideology_plot():
        # Base interactive plot (uses your blue–white–red colorscale)
        fig = plot_ideology_numberline(vote_matrix, chamber="house")

        name_to_mark = highlighted_member.get()
        if name_to_mark:
            # Work on a copy to avoid mutating the original
            vm = vote_matrix.copy()

            # Case-insensitive partial match on the member name
            mask = vm["name"].str.contains(name_to_mark, case=False, na=False)

            if mask.any():
                row = vm[mask].iloc[0]

                # Use the SAME coordinate system as your plot function:
                # x = extremity_z, y = extremity_PC2 (or PC2 if that’s what you use)
                x_val = float(row["extremity_z"])
                y_val = float(row["extremity_PC2"])

                # Add a bright yellow marker on top of the existing trace
                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode="markers+text",
                        marker=dict(
                            size=24,
                            color="#ffd60a",              # bright yellow highlight
                            line=dict(width=2, color="black"),
                            symbol="star"
                        ),
                        textposition="top center",
                        name="Highlighted member",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        return fig

   
    member_plot_fig = reactive.Value(None)
    member_narrative_text = reactive.Value("")   # narrative summary text
    member_bucket_stats = reactive.Value({})     # NEW: per-bucket YES%

    # AUTO-LOAD DEFAULT MEMBER (Cline) ON APP START
    @reactive.effect
    def _initial_member_autoload():
        if member_plot_fig.get() is None:  # Only run once at startup
            default_name = "Cline"  # Default member for the radar chart
            try:
                fig = prompt_for_member_and_plot(default_name)
                member_plot_fig.set(fig)
                # also set default narrative
                narr = get_member_narrative(default_name)
                member_narrative_text.set(narr.get("narrative_text", ""))
                stats = get_member_bucket_stats(default_name)
                member_bucket_stats.set(stats)
            except Exception:
                pass  # Handle errors gracefully (e.g., if the default member is not found)

    # BUTTON BEHAVIOR
    @reactive.effect
    @reactive.event(input.member_go)
    def _run_member_attribute():
        name = clicked_member_name.get()
        if not name:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="Click a suggestion to select a member, then press Generate.",
                x=0.5, y=0.5, showarrow=False
            )
            member_plot_fig.set(fig)
            member_narrative_text.set("")
            member_bucket_stats.set({})   # clear
            return
        try:
            fig = prompt_for_member_and_plot(name)
            member_plot_fig.set(fig)
            narr = get_member_narrative(name)
            member_narrative_text.set(narr.get("narrative_text", ""))
            stats = get_member_bucket_stats(name)
            member_bucket_stats.set(stats)
        except Exception as e:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", x=0.5, y=0.5, showarrow=False)
            member_plot_fig.set(fig)
            member_narrative_text.set(f"Error generating narrative: {e}")
            member_bucket_stats.set({})

    @output
    @render_widget
    def member_plot():
        fig = member_plot_fig.get()  # Get the current radar chart figure
        return fig

    @output
    @render.ui
    def member_narrative_card():
        text = member_narrative_text.get() or "Generate a member plot to see narrative."
        stats = member_bucket_stats.get() or {}
        # Build stats table HTML
        table_rows = ""
        # Order buckets consistently
        bucket_order = [
            "Economy & Budget",
            "Defense & Security",
            "Health & Public Safety",
            "Energy & Environment",
            "Procedure, Ethics & Oversight",
            "Veterans & Services",
        ]
        for lbl in bucket_order:
            if lbl in stats:
                table_rows += f"<tr><td>{lbl}</td><td class='num'>{stats[lbl]:.0f}%</td></tr>"

        html = f"""
        <style>
          .narrative-card {{
              display:flex; flex-direction:column; gap:0.8rem;
          }}
          .narrative-card p {{
              margin: 0; color: #333; font-size: 0.95rem; line-height: 1.45;
          }}
          .stats-table {{
              width: 100%;
              border-collapse: collapse;
              font-size: 0.92rem;
              margin-top: 0.25rem;
              border: 1px solid #e0e0e0;
              border-radius: 10px;
              overflow: hidden;
          }}
          .stats-table thead {{
              background: #f5f5f5;
          }}
          .stats-table th, .stats-table td {{
              padding: 8px 12px;
              border-bottom: 1px solid #eaeaea;
          }}
          .stats-table tbody tr:nth-child(even) {{
              background: #fafafa;
          }}
          .stats-table td.num {{
              text-align: right;
              font-variant-numeric: tabular-nums;
          }}
        </style>
        <div class="narrative-card">
          <p><strong>Narrative summary</strong></p>
          <p>{text}</p>
          <div>
            <p class="legend-label" style="margin-top:0.4rem;"></p>
            <table class="stats-table">
              <thead><tr><th>Issue area</th><th></th></tr></thead>
              <tbody>
                {table_rows if table_rows else "<tr><td colspan='2' style='text-align:center;color:#777;'>No stats available</td></tr>"}
              </tbody>
            </table>
          </div>
        </div>
        """
        return ui.HTML(html)

    kmeans_fig = reactive.Value(None)

    @reactive.effect
    def _initial_kmeans_load():
        if kmeans_fig.get() is None:  # Only run once at startup
            try:
                k = 5
                df = congress_model_user.congress_vote_matrix.copy()
                X = df[["PC1", "PC2"]].to_numpy()
                km = KMeans(n_clusters=k, random_state=0)
                df["cluster"] = km.fit_predict(X)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["PC1"],
                    y=df["PC2"],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=df["cluster"],
                        colorscale="Viridis",
                        opacity=0.9,
                        line=dict(width=0.7, color="black")
                    ),
                    # text=df["name"],
                    hovertemplate="<b>%{text}</b><br>Extremity1: %{x:.2f}<br>Faction: %{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="Extremity (Z-score)",
                    yaxis_title="Intra-party Faction (PC2)",
                    plot_bgcolor="#fafafa",
                    paper_bgcolor="#fafafa",
                    height=550,
                    margin=dict(l=40, r=40, t=60, b=40),
                    autosize=True,
                    width=1400,
                )
                kmeans_fig.set(fig)
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error running KMeans: {e}", showarrow=False)
                kmeans_fig.set(fig)

    @reactive.effect
    @reactive.event(input.kmeans_go)
    def _run_kmeans():
        try:
            k = int(input.k_clusters())
            df = get_knn_pca(congress_model_user.df.copy(), k)
            X = df[["PC1", "PC2"]].to_numpy()
            km = KMeans(n_clusters=k, random_state=0)
            df["cluster"] = km.fit_predict(X)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["PC1"],
                y=df["PC2"],
                mode="markers",
                marker=dict(
                    size=11,
                    color=df["cluster"],
                    colorscale="Viridis",
                    opacity=0.9,
                    line=dict(width=0.7, color="black")
                ),
                # text=df["name"],
                hovertemplate="<b>%{text}</b><br>Extremity1: %{x:.2f}<br>Faction: %{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="Extremity (Z-score)",
                yaxis_title="Intra-party Faction (PC2)",
                plot_bgcolor="#fafafa",
                paper_bgcolor="#fafafa",
                height=550,
                margin=dict(l=40, r=40, t=60, b=40),
                autosize=True,
                width=1400,
            )
            kmeans_fig.set(fig)
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error running KMeans: {e}", showarrow=False)
            kmeans_fig.set(fig)

    
    @output
    @render_widget
    def kmeans_plot():
        import plotly.graph_objects as go
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np
        from sklearn.cluster import KMeans

        idx = current_vote_index.get()
        bills = updated_top_bills_df.reset_index(drop=True)
        vote_matrix = congress_model_user.congress_vote_matrix.copy()
        scaler = congress_model_user.scaler
        pca = congress_model_user.pca

        # Define color mapping and cluster names
        color_map = {
            "Hardline Republicans":     "#8B0000",  # dark red
            "Hardline Democrats":       "#3B74C1",  # dark blue
            "Moderate Left Leaning":    "#ADD8E6",  # light blue
            "Wild Card Voters":         "#4B0082",  # purple
            "Less Extreme Republicans": "#FF7F7F",  # light red
        }

        cluster_order = [
            "Hardline Republicans",
            "Hardline Democrats",
            "Moderate Left Leaning",
            "Wild Card Voters",
            "Less Extreme Republicans"
        ]

        cluster_name_map = {
            0: "Hardline Republicans",
            1: "Moderate Left Leaning",
            2: "Hardline Democrats",
            3: "Wild Card Voters",
            4: "Less Extreme Republicans"
        }

        # Feature columns used during training
        n_features = pca.components_.shape[1]
        feature_cols = vote_matrix.columns[:n_features]

        # Ensure clusters exist
        if "knn_cluster" not in vote_matrix.columns:
            km = KMeans(n_clusters=5, random_state=0)
            vote_matrix["knn_cluster"] = km.fit_predict(vote_matrix[feature_cols])

        # Map labels and colors
        vote_matrix["cluster_label"] = vote_matrix["knn_cluster"].map(cluster_name_map)
        vote_matrix["color"] = vote_matrix["cluster_label"].map(color_map)
        vote_matrix["size"] = 11
        vote_matrix["is_user"] = False

        hover_cols = ["name", "party", "state"]
        if all(col in vote_matrix.columns for col in hover_cols):
            vote_matrix["hover"] = (
                vote_matrix["name"] + " (" + vote_matrix["party"] + ", " + vote_matrix["state"] + ")<br>" +
                "Cluster: " + vote_matrix["cluster_label"]
            )
        else:
            vote_matrix["hover"] = (
                vote_matrix.index.astype(str) + "<br>Cluster: " + vote_matrix["cluster_label"]
            )

        # --- Build Plot ---
        fig = go.Figure()

        for label in cluster_order:
            df = vote_matrix[vote_matrix["cluster_label"] == label]
            fig.add_trace(go.Scatter(
                x=df["PC1"],
                y=df["PC2"],
                mode="markers",
                marker=dict(
                    size=df["size"],
                    color=color_map[label],
                    line=dict(width=0.7, color="black"),
                    opacity=0.9
                ),
                name=label,
                hovertext=df["hover"],
                hoverinfo="text",
                showlegend=True,
            ))

        # --- Add user star if voting complete ---
        if idx >= len(bills):
            votes = user_votes.get()
            user_df = pd.DataFrame(columns=feature_cols)
            user_df.loc["You"] = np.nan
            for bill, val in votes.items():
                if bill in user_df.columns:
                    user_df.loc["You", bill] = float(val) if val != "" else np.nan

            user_df = user_df.apply(lambda row: row.fillna(row.mean()), axis=1)
            if user_df.isna().all(axis=None):
                user_df.loc["You"] = vote_matrix[feature_cols].mean()

            user_scaled = scaler.transform(user_df[feature_cols])
            user_coords = pca.transform(user_scaled)
            user_PC1, user_PC2 = user_coords[0]

            # Predict cluster using KNN
            knn = KNeighborsClassifier(n_neighbors=15)
            knn.fit(vote_matrix[["PC1", "PC2"]], vote_matrix["knn_cluster"])
            user_cluster = int(knn.predict([[user_PC1, user_PC2]])[0])
            user_cluster_name = cluster_name_map.get(user_cluster, f"Cluster {user_cluster}")
            user_color = color_map.get(user_cluster_name, "#D0021B")

            # Add user star with hover
            fig.add_trace(go.Scatter(
                x=[user_PC1],
                y=[user_PC2],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=24,
                    color="yellow",
                    line=dict(width=2, color="black")
                ),
                text=[f"You – {user_cluster_name}"],
                hoverinfo="text",
                showlegend=False
            ))

        # --- Layout ---
        fig.update_layout(
            xaxis_title="PC1 (Political Spectrum)",
            yaxis_title="PC2 (Within-Party Voting Variation)",
            legend_title_text="Factions",
            plot_bgcolor="#fafafa",
            paper_bgcolor="#fafafa",
            height=550,
            width=1400,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig


        
    @output
    @render.ui
    def top10_bills_table():
        # You should load or compute updated_top_bills_df in congress_model_user.py and import it here
        from congress_model_user import updated_top_bills_df

        # Columns to show
        cols = ["title", "influence"]
        df = updated_top_bills_df[cols].copy()
        # df["PC1_loading"] = df["PC1_loading"].map(lambda x: f"{x:.4f}")
        # df["PC2_loading"] = df["PC2_loading"].map(lambda x: f"{x:.4f}")
        df["influence"] = df["influence"].map(lambda x: f"{x:.4f}")

        # Build HTML table
        html = """
        <style>
            .top10-table-container {
                max-height: 520px;
                overflow-y: auto;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                margin-top: 0.75rem;
            }
            .top10-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }
            .top10-table thead {
                background: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 2;
            }
            .top10-table th, .top10-table td {
                padding: 10px 14px;
                border-bottom: 1px solid #e5e5e5;
                text-align: left;
            }
            .top10-table tbody tr:nth-child(even) {
                background: #fafafa;
            }
            .top10-table tbody tr:hover {
                background: #eaeaea;
            }
        </style>
        <div class="top10-table-container">
        <table class="top10-table">
            <thead>
                <tr>
        """
        for c in cols:
            html += f"<th>{c}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            html += "<tr>"
            for c in cols:
                html += f"<td>{row[c]}</td>"
            html += "</tr>"

        html += "</tbody></table></div>"
        return ui.HTML(html)

    @output
    @render.ui
    def user_bill_vote_table():
        idx = current_vote_index.get()
        bills = updated_top_bills_df.reset_index(drop=True)
        if idx >= len(bills):
            html = "<div style='font-size:1.1rem; color:#155724; padding:1.5rem 0;'>Thank you for voting!<br>Your responses have been recorded.</div>"
            return ui.HTML(html)

        row = bills.iloc[idx]
        bill = row["bill_number"]
        title = row["title"]
        summary = row["summary"]
        bill_type = row["bill_type"]

        votes = user_votes.get()
        prev_vote = votes.get(bill, "")

        return ui.div(
            {"style": "max-width:600px; margin:auto;"},
            ui.tags.div(
                ui.p(f"Bill {idx+1} of {len(bills)}", class_="legend-label", style="margin-bottom:0.7rem;"),
            ),
            ui.div(
                {
                    "class": "vote-card",
                    "style": (
                        "margin-bottom:1.2rem; "
                        "padding-bottom:0.5rem; "
                        "display:flex; flex-direction:column;"
                    )
                },
                ui.div(
                    {"class": "vote-card-title"},
                    f"{title} ",
                    ui.span(f"({bill_type})", style="font-size:0.92em; color:#888;"),
                ),
                ui.div(
                    {
                        "class": "vote-card-summary",
                        "style": (
                            "max-height:320px; overflow-y:auto; background:#f4f4f4; "
                            "border-radius:0.7rem; padding:0.7rem; margin-bottom:0.7rem;"
                        )
                    },
                    summary
                ),
                # --- Button row: radio buttons and Next button side by side ---
                ui.div(
                    {
                        "class": "vote-card-footer",
                        "style": (
                            "display:flex; "
                            "flex-direction:row; "
                            "align-items:center; "
                            "justify-content:space-between; "
                            "gap:1.2rem; "
                            "width:100%; "
                            "margin-top:0.5rem;"
                        )
                    },
                    ui.div(
                        ui.input_radio_buttons(
                            f"vote_{bill}",
                            "Your Vote:",
                            choices={"1": "Yes", "0": "No", "": "Skip"},
                            selected=prev_vote,
                            inline=True,
                        ),
                        style="flex:1;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "next_vote_btn",
                            "Next",
                            class_="pill-button-primary",
                            style="width:220px; padding:0.6rem 1rem;"
                            
                        ),
                        style="flex:none;"
                    ),
                ),
            ),
        )

    @output
    @render.ui
    def similar_legislators():
        # Get the nearest legislators DataFrame
        df = nearest_legislators_df.get()

        if df is None or df.empty:
            return ui.HTML("<p>No similar members found. Please complete your votes.</p>")

        # Columns to show
        cols = ["name", "party", "cluster_label", "distance_to_user"]
        df = df[cols].copy()
        df = df.sort_values("distance_to_user", ascending=False)
        df["distance_to_user"] = df["distance_to_user"].map(lambda x: f"{x:.4f}")

        # Build HTML table
        html = """
        <style>
            .similar-members-container {
                display: flex;
                flex-direction: column;
                height: 100%; /* Make the container take full height */
            }
            .similar-table-container {
                flex-grow: 1; /* Allow the table to stretch */
                overflow-y: auto;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                margin-top: 0.75rem;
            }
            .similar-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                height: 100%; /* Ensure the table stretches */
            }
            .similar-table thead {
                background: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 2;
            }
            .similar-table th, .similar-table td {
                padding: 10px 14px;
                border-bottom: 1px solid #e5e5e5;
                text-align: left;
            }
            .similar-table tbody tr:nth-child(even) {
                background: #fafafa;
            }
            .similar-table tbody tr:hover {
                background: #eaeaea;
            }
        </style>
        <div class="similar-members-container">
            <div class="similar-table-container">
                <table class="similar-table">
                    <thead>
                        <tr>
        """
        for c in cols:
            html += f"<th>{c}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            html += "<tr>"
            for c in cols:
                html += f"<td>{row[c]}</td>"
            html += "</tr>"

        html += "</tbody></table></div></div>"
        return ui.HTML(html)

    current_vote_index = reactive.Value(0)
    user_votes = reactive.Value({})

    @reactive.effect
    @reactive.event(input.next_vote_btn)
    def _advance_vote():
        idx = current_vote_index.get()
        bills = updated_top_bills_df.reset_index(drop=True)
        if idx < len(bills):
            bill = bills.iloc[idx]["bill_number"]
            val = input[f"vote_{bill}"]()
            votes = user_votes.get().copy()
            votes[bill] = val
            user_votes.set(votes)
            current_vote_index.set(idx + 1)

            # Trigger computation of nearest legislators if this is the last vote
            if idx + 1 == len(bills):
                compute_nearest_legislators()

    @reactive.effect
    @reactive.event(input.submit_votes)
    def _finish_voting():
        bills = updated_top_bills_df.reset_index(drop=True)
        current_vote_index.set(len(bills))

    # Add this helper function near the top of your server (outside any output):
    def get_user_pca_coords(user_votes, vote_matrix, scaler, pca):
        import numpy as np
        import pandas as pd
        n_features = pca.components_.shape[1]
        feature_cols = vote_matrix.columns[:n_features]
        user_df = pd.DataFrame(columns=feature_cols)
        user_df.loc["You"] = np.nan
        for bill, val in user_votes.items():
            if bill in user_df.columns:
                user_df.loc["You", bill] = float(val) if val != "" else np.nan
        user_df = user_df.apply(lambda row: row.fillna(row.mean()), axis=1)
        if user_df.isna().all(axis=None):
            user_df.loc["You"] = vote_matrix[feature_cols].mean()
        user_scaled = scaler.transform(user_df[feature_cols])
        user_coords = pca.transform(user_scaled)
        return user_coords[0, 0], user_coords[0, 1]

    # Compute the nearest legislators
    nearest_legislators_df = reactive.Value(None)

    # Shared cluster name mapping used across components
    cluster_name_map = {
        0: "Hardline Republicans",
        1: "Moderate Left Leaning",
        2: "Hardline Democrats",
        3: "Wild Card Voters",
        4: "Less Extreme Republicans"
    }

    def compute_nearest_legislators():
        # Get the user's PCA coordinates dynamically
        user_votes_dict = user_votes.get()
        user_PC1, user_PC2 = get_user_pca_coords(
            user_votes=user_votes_dict,
            vote_matrix=congress_model_user.congress_vote_matrix,
            scaler=congress_model_user.scaler,
            pca=congress_model_user.pca,
        )

        # Compute the nearest legislators
        nearest_legislators = get_nearest_legislators(
            vote_matrix=congress_model_user.congress_vote_matrix,
            user_PC1=user_PC1,
            user_PC2=user_PC2,
            cluster_name_map=cluster_name_map,
            df=congress_model_user.df,
        )

        # Store the result in a reactive value
        nearest_legislators_df.set(nearest_legislators)
        
    @reactive.effect
    @reactive.event(input.highlight_name_input)
    def _update_suggestions_for_highlight():
        query = input.highlight_name_input().strip().lower()
        if query:
            suggestions = [name for name in LEGISLATOR_NAMES if query in name.lower()]
            highlight_suggestions.set(suggestions[:10])  # Limit to top 10 suggestions
        else:
            highlight_suggestions.set([])
    
    @reactive.effect
    @reactive.event(input.member_query_input)
    def _update_suggestions_for_member():
        query = (input.member_query_input() or "").strip().lower()
        if query:
            suggestions = [name for name in LEGISLATOR_NAMES if query in name.lower()]
            member_suggestions.set(suggestions[:10])
        else:
            member_suggestions.set([])
    
    @output
    @render.ui
    def highlight_name_suggestions():
        suggestions = highlight_suggestions.get()
        if not suggestions:
            return ui.HTML("")  # No suggestions to show

        # Build the dropdown list
        html = "<ul style='list-style: none; padding: 0; margin: 0;'>"
        for name in suggestions:
            html += (
                f"<li style='padding: 8px; cursor: pointer; transition: background-color 0.2s;' "
                f"onmouseover='this.style.backgroundColor=\"#e0e0e0\";' "
                f"onmouseout='this.style.backgroundColor=\"white\";' "
                # 1) Put clicked name into the text box
                f"onclick='document.getElementById(\"highlight_name_input\").value = \"{name}\"; "
                # 2) Emit a Shiny input event carrying the clicked name
                f"Shiny.setInputValue(\"clicked_highlight_name\", \"{name}\", {{priority: \"event\"}}); "
                # 3) Close the dropdown
                f"document.getElementById(\"highlight_name_suggestions\").innerHTML = \"\";'>{name}</li>"
            )
        html += "</ul>"
        return ui.HTML(html)

    @output
    @render.ui
    def member_query_suggestions():
        suggestions = member_suggestions.get()
        if not suggestions:
            return ui.HTML("")  # No suggestions to show

        # Build the dropdown list
        html = "<ul style='list-style: none; padding: 0; margin: 0;'>"
        for name in suggestions:
            html += (
                f"<li style='padding: 8px; cursor: pointer; transition: background-color 0.2s;' "
                f"onmouseover='this.style.backgroundColor=\"#e0e0e0\";' "
                f"onmouseout='this.style.backgroundColor=\"white\";' "
                # 1) Put clicked name into the text box
                f"onclick='document.getElementById(\"member_query_input\").value = \"{name}\"; "
                # 2) Emit a Shiny input event carrying the clicked name
                f"Shiny.setInputValue(\"clicked_member_name\", \"{name}\", {{priority: \"event\"}}); "
                # 3) Close the dropdown
                f"document.getElementById(\"member_query_suggestions\").innerHTML = \"\";'>{name}</li>"
            )
        html += "</ul>"
        return ui.HTML(html)

# ---------------------------------------------------------------------------
# App object and entrypoint
# ---------------------------------------------------------------------------

app = App(app_ui, server)

# Extract legislator names from the dataset
LEGISLATOR_NAMES = sorted(congress_model_user.df["name"].dropna().unique().tolist())
