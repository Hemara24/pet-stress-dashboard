import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from dash.dependencies import Input, Output

# Load cleaned data
df = pd.read_csv("pet_stress_processed.csv")

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # REQUIRED for Render

app.layout = html.Div([

    # Header with gradient background - REDUCED PADDING
    html.Div([
        html.H1("🐾 The Impact of Pet Interaction on Stress Among Pet Owners in the Western Province",
                style={
                    "textAlign": "center",
                    "color": "white",
                    "fontFamily": "Arial, sans-serif",
                    "fontSize": "28px",
                    "fontWeight": "bold",
                    "margin": "0",
                    "padding": "15px 20px 10px 20px",
                    "textShadow": "2px 2px 4px rgba(0,0,0,0.3)",
                    "lineHeight": "1.3"
                })
    ], style={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "marginBottom": "0px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
    }),

    # Main container with sidebar and graphs
    html.Div([
        
        # Left Sidebar - Filters - REDUCED PADDING
        html.Div([
            html.Div([
                html.H3("🔍 Filters",
                        style={
                            "textAlign": "center",
                            "marginBottom": "15px",
                            "color": "#667eea",
                            "fontFamily": "Arial, sans-serif",
                            "fontWeight": "bold",
                            "fontSize": "16px"
                        }),
                
                # Pet Type Filter
                html.Div([
                    html.Label("🐶 Pet Type",
                               style={
                                   "fontWeight": "bold",
                                   "marginBottom": "5px",
                                   "display": "block",
                                   "color": "#2d3748",
                                   "fontSize": "13px"
                               }),
                    dcc.Dropdown(
                        id="pet_type_filter",
                        options=[{"label": p, "value": p} for p in df["pet_type"].unique()],
                        value=df["pet_type"].unique().tolist(),
                        multi=True,
                        style={"marginBottom": "12px", "fontSize": "12px"}
                    ),
                ], style={"marginBottom": "12px"}),
                
                # Stress Level Filter
                html.Div([
                    html.Label("😰 Stress Level",
                               style={
                                   "fontWeight": "bold",
                                   "marginBottom": "5px",
                                   "display": "block",
                                   "color": "#2d3748",
                                   "fontSize": "13px"
                               }),
                    dcc.Checklist(
                        id="stress_filter",
                        options=[{"label": s, "value": s} for s in df["stress_level"].unique()],
                        value=df["stress_level"].unique().tolist(),
                        style={"marginBottom": "8px", "fontSize": "12px"},
                        labelStyle={"display": "block", "marginBottom": "5px"}
                    ),
                ], style={"marginBottom": "12px"}),

                # Age Range Filter
                html.Div([
                    html.Label("👤 Age Range",
                               style={
                                   "fontWeight": "bold",
                                   "marginBottom": "5px",
                                   "display": "block",
                                   "color": "#2d3748",
                                   "fontSize": "13px"
                               }),
                    dcc.RangeSlider(
                        id="age_filter",
                        min=df["owner_age"].min(),
                        max=df["owner_age"].max(),
                        step=1,
                        value=[df["owner_age"].min(), df["owner_age"].max()],
                        marks={int(df["owner_age"].min()): str(int(df["owner_age"].min())), 
                               int(df["owner_age"].max()): str(int(df["owner_age"].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={"marginBottom": "15px"}),
                
                # Filter Status - REDUCED PADDING
                html.Div(id="filter_status", style={
                    "padding": "10px",
                    "backgroundColor": "#edf2f7",
                    "marginTop": "15px",
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "borderRadius": "8px",
                    "wordWrap": "break-word",
                    "border": "2px solid #667eea",
                    "color": "#2d3748"
                }),
                
            ], style={
                "backgroundColor": "white",
                "padding": "15px",
                "borderRadius": "12px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
            })
            
        ], style={
            "width": "20%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "10px",
            "backgroundColor": "#f7fafc"
        }),

        # Right Side - Graphs in 3x2 Grid - REDUCED HEIGHT & PADDING
        html.Div([
            # Row 1
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_stress_dist", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px",
                        "transition": "transform 0.2s"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_pet_stress", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_ccas_stress", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
            ], style={"marginBottom": "10px"}),
            
            # Row 2
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_scatter", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_regression", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id="filtered_bar", style={"height": "260px"})
                    ], style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "padding": "10px"
                    })
                ], style={"width": "32%", "display": "inline-block", "padding": "5px", "verticalAlign": "top"}),
            ])
            
        ], style={
            "width": "78%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "10px",
            "backgroundColor": "#f7fafc"
        })
        
    ], style={"display": "flex", "minHeight": "100vh"})

], style={"backgroundColor": "#f7fafc", "fontFamily": "Arial, sans-serif"})

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
    
    # Create filter status message with emojis
    filter_msg = html.Div([
        html.Div(f"📊 {len(filtered_df)} / {len(df)} Participants", style={"fontWeight": "bold", "fontSize": "13px", "marginBottom": "6px"}),
        html.Div(f"🐾 {', '.join(map(str, pet_types[:3]))}{' ...' if len(pet_types) > 3 else ''}", style={"fontSize": "10px", "marginBottom": "3px"}),
        html.Div(f"😊 {', '.join(map(str, stress_levels))}", style={"fontSize": "10px", "marginBottom": "3px"}),
        html.Div(f"👤 Age: {age_range[0]}-{age_range[1]}", style={"fontSize": "10px"})
    ])

    # 1. Stress Level Distribution
    stress_counts = filtered_df["stress_level"].value_counts()
    chi2_stress, p_val_stress = stats.chisquare(stress_counts)
    
    stress_dist_fig = px.bar(
        filtered_df["stress_level"].value_counts().reset_index(),
        x="stress_level",
        y="count",
        title=f"<b>Stress Distribution</b><br><sub>χ²={chi2_stress:.2f}, p={p_val_stress:.4f}</sub>",
        labels={"stress_level": "Stress Level", "count": "Count"},
        color="stress_level",
        color_discrete_map={"Low": "#48bb78", "Moderate": "#4299e1", "High": "#f56565"}
    )
    stress_dist_fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=10)
    )

    # 2. Stress by Pet Type
    contingency_table = pd.crosstab(filtered_df["pet_type"], filtered_df["stress_level"])
    chi2_pet, p_val_pet, dof, expected = stats.chi2_contingency(contingency_table)
    
    pet_stress_fig = px.bar(
        filtered_df,
        x="pet_type",
        color="stress_level",
        barmode="group",
        title=f"<b>Stress by Pet Type</b><br><sub>χ²={chi2_pet:.2f}, p={p_val_pet:.4f}</sub>",
        labels={"pet_type": "Pet Type", "stress_level": "Stress"},
        color_discrete_map={"Low": "#48bb78", "Moderate": "#4299e1", "High": "#f56565"}
    )
    pet_stress_fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=10)
    )

    # 3. Stress by Attachment
    contingency_ccas = pd.crosstab(filtered_df["comfort_level"], filtered_df["stress_level"])
    chi2_ccas, p_val_ccas, dof_ccas, expected_ccas = stats.chi2_contingency(contingency_ccas)
    
    ccas_stress_fig = px.bar(
        filtered_df,
        x="comfort_level",
        color="stress_level",
        barmode="stack",
        title=f"<b>Stress by Attachment</b><br><sub>χ²={chi2_ccas:.2f}, p={p_val_ccas:.4f}</sub>",
        labels={"comfort_level": "Comfort Level", "stress_level": "Stress"},
        color_discrete_map={"Low": "#48bb78", "Moderate": "#4299e1", "High": "#f56565"}
    )
    ccas_stress_fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=10)
    )

    # 4. Scatter plot
    if len(filtered_df) > 2:
        corr_coef, p_val_corr = stats.pearsonr(filtered_df["ccas_score"], filtered_df["pss_score"])
        r_squared_scatter = corr_coef ** 2
        
        scatter_fig = px.scatter(
            filtered_df,
            x="ccas_score",
            y="pss_score",
            color="pet_type",
            title=f"<b>CCAS vs PSS</b><br><sub>r={corr_coef:.3f}, R²={r_squared_scatter:.3f}, p={p_val_corr:.4f}</sub>",
            labels={"ccas_score": "CCAS Score", "pss_score": "PSS Score"}
        )
        scatter_fig.update_layout(
            margin=dict(l=30, r=30, t=60, b=30),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9)),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=10)
        )
    else:
        scatter_fig = px.scatter(title="<b>Insufficient Data</b>")
        scatter_fig.update_layout(margin=dict(l=30, r=30, t=60, b=30))

    # 5. Regression
    if len(filtered_df) > 2:
        X = sm.add_constant(filtered_df["ccas_score"])
        model = sm.OLS(filtered_df["pss_score"], X).fit()
        x_vals = np.linspace(filtered_df["ccas_score"].min(), filtered_df["ccas_score"].max(), 100)
        y_vals = model.predict(sm.add_constant(x_vals))

        regression_fig = go.Figure()
        regression_fig.add_trace(
            go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Regression",
                      line=dict(color="#667eea", width=4))
        )
        regression_fig.update_layout(
            title=f"<b>Regression Line</b><br><sub>R²={model.rsquared:.3f}, p={model.f_pvalue:.4f}, β={model.params[1]:.3f}</sub>",
            xaxis_title="CCAS Score",
            yaxis_title="Predicted PSS",
            margin=dict(l=30, r=30, t=60, b=30),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=10)
        )
    else:
        regression_fig = go.Figure()
        regression_fig.update_layout(title="<b>Insufficient Data</b>", margin=dict(l=30, r=30, t=60, b=30))

    # 6. Mean Stress
    grouped_data = filtered_df.groupby("comfort_level", as_index=False)["pss_score"].mean()
    
    if len(filtered_df["comfort_level"].unique()) > 1:
        groups = [filtered_df[filtered_df["comfort_level"] == level]["pss_score"].values 
                  for level in filtered_df["comfort_level"].unique()]
        f_stat, p_val_anova = stats.f_oneway(*groups)
        
        grand_mean = filtered_df["pss_score"].mean()
        ss_between = sum(len(filtered_df[filtered_df["comfort_level"] == level]) * 
                        (filtered_df[filtered_df["comfort_level"] == level]["pss_score"].mean() - grand_mean)**2 
                        for level in filtered_df["comfort_level"].unique())
        ss_total = sum((filtered_df["pss_score"] - grand_mean)**2)
        r_squared_anova = ss_between / ss_total if ss_total > 0 else 0
        
        bar_fig = px.bar(
            grouped_data,
            x="comfort_level",
            y="pss_score",
            title=f"<b>Mean Stress by Comfort</b><br><sub>F={f_stat:.2f}, η²={r_squared_anova:.3f}, p={p_val_anova:.4f}</sub>",
            labels={"comfort_level": "Comfort Level", "pss_score": "Mean PSS"},
            color="pss_score",
            color_continuous_scale=[[0, "#48bb78"], [0.5, "#ecc94b"], [1, "#f56565"]]
        )
        bar_fig.update_layout(
            margin=dict(l=30, r=30, t=60, b=30),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=10)
        )
    else:
        bar_fig = px.bar(grouped_data, x="comfort_level", y="pss_score", title="<b>Mean Stress</b>")
        bar_fig.update_layout(margin=dict(l=30, r=30, t=60, b=30))

    return filter_msg, stress_dist_fig, pet_stress_fig, ccas_stress_fig, scatter_fig, regression_fig, bar_fig

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)