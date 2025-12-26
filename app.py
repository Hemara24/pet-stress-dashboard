import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# Load cleaned data
df = pd.read_csv("pet_stress_processed.csv")

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # REQUIRED for Render

# 1. CCAS vs PSS Relationship (Scatter plot)
scatter_fig = px.scatter(
    df,
    x="ccas_score",
    y="pss_score",
    title="Scatter Plot: CCAS Score vs PSS Score",
    labels={
        "ccas_score": "CCAS Score",
        "pss_score": "PSS Score"
    }
)


# 2. Mean PSS by comfort level  (Bar chart)
bar_fig = px.bar(
    df.groupby("comfort_level", as_index=False)["pss_score"].mean(),
    x="comfort_level",
    y="pss_score",
    title="Mean Stress Level by CCAS Category",
    labels={
        "comfort_level": "Comfort Level",
        "pss_score": "Mean PSS Score"
    }
)

# 3. Regression Trendline: CCAS vs PSS
X = df["ccas_score"]
y = df["pss_score"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

x_range = np.linspace(X.min(), X.max(), 100)
y_pred = model.predict(sm.add_constant(x_range))

regression_fig = go.Figure()
regression_fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode="lines",
        name="Regression Line"
    )
)

regression_fig.update_layout(
    title="Regression Trendline: CCAS vs PSS",
    xaxis_title="CCAS Score",
    yaxis_title="Predicted PSS Score"
)

# 4. Overall Stress Level Distribution (Bar chart)
stress_dist_fig = px.bar(
    df["stress_level"].value_counts().reset_index(),
    x="stress_level",
    y="count",
    title="Overall Stress Level Distribution",
    labels={
        "stress_level": "Stress Level",
        "count": "Number of Participants"
    }
)

# 5. Stress Level by Pet Type (Grouped Bar chart)
pet_stress_fig = px.bar(
    df,
    x="pet_type",
    color="stress_level",
    barmode="group",
    title="Stress Level by Pet Type",
    labels={
        "pet_type": "Pet Type",
        "stress_level": "Stress Level"
    }
)

# 6. Stress Level by Comfort Level (Stacked Bar chart)
ccas_stress_fig = px.bar(
    df,
    x="comfort_level",
    color="stress_level",
    barmode="stack",
    title="Stress Levels by Pet Attachment (CCAS)",
    labels={
        "comfort_level": "Comfort Level",
        "stress_level": "Stress Level"
    }
)

app.layout = html.Div([

    html.H1("Impact of Pet Interaction on Stress",
            style={"textAlign": "center"}),

    # Filters Section
    html.Div([
        html.H3("Filter Options", style={"textAlign": "center"}),
        
        html.Label("Select Pet Type:"),
        dcc.Dropdown(
            id="pet_type_filter",
            options=[{"label": p, "value": p} for p in df["pet_type"].unique()],
            value=df["pet_type"].unique().tolist(),
            multi=True
        ),

        html.Br(),

        html.Label("Select Stress Level:"),
        dcc.Checklist(
            id="stress_filter",
            options=[{"label": s, "value": s} for s in df["stress_level"].unique()],
            value=df["stress_level"].unique().tolist(),
            inline=True
        ),

        html.Br(),

        html.Label("Owner Age Range:"),
        dcc.RangeSlider(
            id="age_filter",
            min=df["owner_age"].min(),
            max=df["owner_age"].max(),
            step=1,
            value=[df["owner_age"].min(), df["owner_age"].max()],
            marks={i: str(i) for i in range(int(df["owner_age"].min()),
                                             int(df["owner_age"].max())+1, 5)}
        )

    ], style={"padding": "20px", "backgroundColor": "#f0f0f0", "marginBottom": "20px"}),

    # Filter Status Display
    html.Div(id="filter_status", style={
        "padding": "10px",
        "backgroundColor": "#e3f2fd",
        "marginBottom": "20px",
        "fontSize": "16px",
        "fontWeight": "bold",
        "textAlign": "center"
    }),

    # All Dynamic Filtered Visualizations - Logical Order
    html.H2("1. Stress Level Distribution"),
    dcc.Graph(id="filtered_stress_dist"),

    html.H2("2. Stress Levels by Pet Type"),
    dcc.Graph(id="filtered_pet_stress"),

    html.H2("3. Stress Levels by Pet Attachment (CCAS)"),
    dcc.Graph(id="filtered_ccas_stress"),

    html.H2("4. Scatter Plot: CCAS vs PSS"),
    dcc.Graph(id="filtered_scatter"),

    html.H2("5. Regression Trendline: CCAS → PSS"),
    dcc.Graph(id="filtered_regression"),

    html.H2("6. Mean Stress by Comfort Level"),
    dcc.Graph(id="filtered_bar")

])

from dash.dependencies import Input, Output

@app.callback(
    Output("filter_status", "children"),
    Output("filtered_stress_dist", "figure"),
    Output("filtered_pet_stress", "figure"),
    Output("filtered_ccas_stress", "figure"),
    Output("filtered_scatter", "figure"),
    Output("filtered_regression", "figure"),
    Output("filtered_bar", "figure"),
    Input("pet_type_filter", "value"),
    Input("stress_filter", "value"),
    Input("age_filter", "value")
)
def update_charts(pet_types, stress_levels, age_range):
    # Handle None or empty values
    if not pet_types:
        pet_types = df["pet_type"].unique().tolist()
    if not stress_levels:
        stress_levels = df["stress_level"].unique().tolist()
    if not age_range:
        age_range = [df["owner_age"].min(), df["owner_age"].max()]
    
    # Filter data
    filtered_df = df[
        (df["pet_type"].isin(pet_types)) &
        (df["stress_level"].isin(stress_levels)) &
        (df["owner_age"].between(age_range[0], age_range[1]))
    ]
    
    print(f"Filtering: Pet Types={pet_types}, Stress={stress_levels}, Age={age_range}")
    print(f"Filtered data size: {len(filtered_df)} rows out of {len(df)} total rows")
    
    # Create filter status message
    filter_msg = f"Showing {len(filtered_df)} of {len(df)} participants | Pet: {', '.join(map(str, pet_types))} | Stress: {', '.join(map(str, stress_levels))} | Age: {age_range[0]}-{age_range[1]}"

    # 1. Stress Level Distribution
    stress_dist_fig = px.bar(
        filtered_df["stress_level"].value_counts().reset_index(),
        x="stress_level",
        y="count",
        title=f"Overall Stress Level Distribution ({len(filtered_df)} participants)",
        labels={
            "stress_level": "Stress Level",
            "count": "Number of Participants"
        },
        color="stress_level",
        color_discrete_map={"Low": "#00d97e", "Moderate": "#4299e1", "High": "#f56565"}
    )

    # 2. Stress Level by Pet Type
    pet_stress_fig = px.bar(
        filtered_df,
        x="pet_type",
        color="stress_level",
        barmode="group",
        title=f"Stress Level by Pet Type ({len(filtered_df)} participants)",
        labels={
            "pet_type": "Pet Type",
            "stress_level": "Stress Level"
        },
        color_discrete_map={"Low": "#00d97e", "Moderate": "#4299e1", "High": "#f56565"}
    )

    # 3. Stress Level by Comfort Level (CCAS)
    ccas_stress_fig = px.bar(
        filtered_df,
        x="comfort_level",
        color="stress_level",
        barmode="stack",
        title=f"Stress Levels by Pet Attachment ({len(filtered_df)} participants)",
        labels={
            "comfort_level": "Comfort Level (CCAS)",
            "stress_level": "Stress Level"
        },
        color_discrete_map={"Low": "#00d97e", "Moderate": "#4299e1", "High": "#f56565"}
    )

    # 4. Scatter plot - CCAS vs PSS
    scatter_fig = px.scatter(
        filtered_df,
        x="ccas_score",
        y="pss_score",
        color="pet_type",
        title=f"Scatter Plot: CCAS Score vs PSS Score ({len(filtered_df)} participants)",
        labels={"ccas_score": "CCAS Score (Pet Attachment)", "pss_score": "PSS Score (Stress)"}
    )

    # 5. Regression trendline
    if len(filtered_df) > 2:
        X = sm.add_constant(filtered_df["ccas_score"])
        model = sm.OLS(filtered_df["pss_score"], X).fit()
        x_vals = np.linspace(filtered_df["ccas_score"].min(),
                              filtered_df["ccas_score"].max(), 100)
        y_vals = model.predict(sm.add_constant(x_vals))

        regression_fig = go.Figure()
        regression_fig.add_trace(
            go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Regression Line", line=dict(color="red", width=3))
        )
        regression_fig.update_layout(
            title=f"Regression Trendline: CCAS → PSS ({len(filtered_df)} participants)",
            xaxis_title="CCAS Score (Pet Attachment)",
            yaxis_title="Predicted PSS Score (Stress)"
        )
    else:
        regression_fig = go.Figure()
        regression_fig.update_layout(title="Not enough data for regression (need at least 3 points)")

    # 6. Bar chart - Mean stress by comfort level
    grouped_data = filtered_df.groupby("comfort_level", as_index=False)["pss_score"].mean()
    bar_fig = px.bar(
        grouped_data,
        x="comfort_level",
        y="pss_score",
        title=f"Mean Stress Level by CCAS Category ({len(filtered_df)} participants)",
        labels={"comfort_level": "Comfort Level (CCAS)", "pss_score": "Mean PSS Score"},
        color="pss_score",
        color_continuous_scale="RdYlGn_r"
    )

    return filter_msg, stress_dist_fig, pet_stress_fig, ccas_stress_fig, scatter_fig, regression_fig, bar_fig

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
