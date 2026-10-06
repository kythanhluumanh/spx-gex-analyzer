import plotly.graph_objects as go

def create_gex_chart(df, spot_price, call_wall, put_wall, flip_level, symbol_name="SPX", is_etf=False):
    fig = go.Figure()

    # Dynamic decimal formatting based on symbol type
    fmt = ".2f" if is_etf else ".0f"

    # 1. Format Strike Y-axis labels to highlight Vol/OI Spikes
    if 'is_spike' in df.columns:
        y_labels = [
            f"🔥 {row['strike']:{fmt}}" if row['is_spike'] else f"{row['strike']:{fmt}}"
            for _, row in df.iterrows()
        ]
    else:
        y_labels = [f"{s:{fmt}}" for s in df['strike']]

    # 2. Add Call GEX Bars (Green)
    fig.add_trace(go.Bar(
        y=y_labels,
        x=df['call_gex'],
        name='Call GEX ($)',
        orientation='h',
        marker=dict(color='#26a69a', opacity=0.85)
    ))

    # 3. Add Put GEX Bars (Red)
    fig.add_trace(go.Bar(
        y=y_labels,
        x=df['put_gex'],
        name='Put GEX ($)',
        orientation='h',
        marker=dict(color='#ef5350', opacity=0.85)
    ))

    # 4. Highlight Key Levels
    fig.add_hline(
        y=spot_price, 
        line_dash="dash", 
        line_color="#29b6f6", 
        annotation_text=f" Spot: {spot_price:.2f}", 
        annotation_position="top right"
    )
    fig.add_hline(
        y=call_wall, 
        line_dash="dot", 
        line_color="#66bb6a", 
        annotation_text=f" Call Wall: {call_wall:{fmt}}", 
        annotation_position="top right"
    )
    fig.add_hline(
        y=put_wall, 
        line_dash="dot", 
        line_color="#ef5350", 
        annotation_text=f" Put Wall: {put_wall:{fmt}}", 
        annotation_position="bottom right"
    )
    fig.add_hline(
        y=flip_level, 
        line_dash="dashdot", 
        line_color="#ab47bc", 
        annotation_text=f" Gamma Flip: {flip_level:{fmt}}", 
        annotation_position="bottom left"
    )

    # 5. Chart Layout & Theme
    fig.update_layout(
        title=f"<b>{symbol_name} 0-DTE Net Gamma Exposure Profile</b>",
        xaxis_title="Net Gamma Exposure ($ GEX)",
        yaxis_title="Strike Price",
        barmode='relative',
        height=750,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig