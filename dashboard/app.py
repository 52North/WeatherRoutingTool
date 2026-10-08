import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
from utils import load_route_data, get_summary_metrics
import json

# Page config
st.set_page_config(
    page_title="Weather Routing Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Design System: Japanese Bento Box (Premium Dark Mode) ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Reset & Base Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #050505; /* Deepest black */
        color: #E0E0E0;
    }
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Bento Card Container (for pure HTML content) */
    .bento-card {
        background-color: #121212;
        border: 1px solid #2A2A2A;
        border-radius: 20px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    /* Simulate Bento Card for Streamlit Components (Map/Charts) */
    /* Target folium iframe container */
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 20px;
        border: 1px solid #2A2A2A;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Target Plotly charts */
    [data-testid="stPlotlyChart"] {
        background-color: #121212;
        border: 1px solid #2A2A2A;
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        color: #888888;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }

    .metric-unit {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 500;
        margin-left: 4px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #050505; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def bento_metric(label, value, unit="", icon=""):
    """
    Renders a Bento-style metric card using HTML.
    """
    st.markdown(f"""
    <div class="bento-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">
            {value}<span class="metric-unit">{unit}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚓ Navigation")
    uploaded_file = st.file_uploader("Upload Route (GeoJSON)", type=['json', 'geojson'])
    st.markdown("---")
    st.caption("Weather Routing Tool v2.0")

# --- Main Layout ---

# Title Section
st.markdown("<h2 style='margin-bottom: 30px; font-weight: 600;'>Weather Routing Dashboard</h2>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load Data
        df, gdf, raw_json = load_route_data(uploaded_file)
        metrics = get_summary_metrics(df)
        
        # --- Row 1: Key Metrics ---
        c1, c2, c3, c4 = st.columns(4)
        with c1: bento_metric("Total Fuel", f"{metrics['total_fuel_tons']:.2f}", "tons", "⛽")
        with c2: 
            time_str = metrics['total_time'].split('.')[0]
            bento_metric("Total Time", time_str, "", "⏱️")
        with c3: bento_metric("Avg Speed", f"{metrics['avg_speed_knots']:.1f}", "kn", "🚀")
        with c4: bento_metric("Waypoints", f"{len(df)}", "pts", "📍")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # --- Row 2: Map & Route Info ---
        m1, m2 = st.columns([3, 1])
        
        with m1:
            # Map Component (Directly rendered, styled via CSS)
            if not df.empty:
                avg_lat = df['latitude'].mean()
                avg_lon = df['longitude'].mean()
                m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles='CartoDB dark_matter', control_scale=True)
                
                points = list(zip(df['latitude'], df['longitude']))
                folium.PolyLine(points, color="#4A90E2", weight=3, opacity=0.9).add_to(m)
                folium.Marker(points[0], popup="Start", icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)
                folium.Marker(points[-1], popup="End", icon=folium.Icon(color="red", icon="stop", prefix='fa')).add_to(m)
                
                st_folium(m, width="100%", height=500, key="main_map")
            
        with m2:
            # Route Details Side Panel
            st.markdown(f"""
<div class="bento-card" style="height: 500px; justify-content: flex-start; gap: 20px;">
<div>
<div class="metric-label">📋 Route Config</div>
<div style="margin-top: 10px; font-size: 0.9rem; color: #ccc;">
<p style="margin-bottom: 5px;"><strong>Type:</strong> {raw_json.get('route type', 'Optimal')}</p>
<p style="margin-bottom: 5px;"><strong>Start:</strong> {metrics['start_time'].strftime('%m-%d %H:%M')}</p>
<p style="margin-bottom: 5px;"><strong>End:</strong> {metrics['end_time'].strftime('%m-%d %H:%M')}</p>
</div>
</div>
<hr style="border-color: #333; width: 100%; margin: 0;">
<div>
<div class="metric-label">Condition Summary</div>
<div style="margin-top: 10px; font-size: 0.9rem; color: #ccc;">
<p style="margin-bottom: 5px;"><strong>Max Wave:</strong> {df['wave_height'].max():.1f} m</p>
<p style="margin-bottom: 5px;"><strong>Max Wind:</strong> {df['u_wind_speed'].max():.1f} m/s</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # --- Row 3: Charts ---
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown('<div class="metric-label" style="margin-bottom: 10px;">⛽ Fuel Profile</div>', unsafe_allow_html=True)
            clean_df = df[df['fuel_consumption'] > -99]
            if not clean_df.empty:
                fig_fuel = px.area(clean_df, x="time", y="fuel_consumption", 
                                  color_discrete_sequence=["#FF5252"])
                fig_fuel.update_layout(
                    plot_bgcolor="#121212", 
                    paper_bgcolor="#121212",
                    font_color="#A0A0A0",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=350,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#333", title=None)
                )
                st.plotly_chart(fig_fuel, use_container_width=True)

        with g2:
            st.markdown('<div class="metric-label" style="margin-bottom: 10px;">🌊 Speed Profile</div>', unsafe_allow_html=True)
            clean_speed = df[df['speed'] > -99]
            if not clean_speed.empty:
                clean_speed['speed_knots'] = clean_speed['speed'] * 1.94384
                fig_speed = px.line(clean_speed, x="time", y="speed_knots",
                                   color_discrete_sequence=["#00E5FF"]) 
                fig_speed.update_layout(
                    plot_bgcolor="#121212", 
                    paper_bgcolor="#121212",
                    font_color="#A0A0A0",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=350,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#333", title=None)
                )
                st.plotly_chart(fig_speed, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading route: {str(e)}")

else:
    # --- Empty State (Landing) ---
    st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; height: 60vh; text-align: center;">
<div>
<div style="font-size: 4rem; margin-bottom: 20px;">🌊</div>
<h1 style="font-weight: 700; margin-bottom: 10px;">Start Your Voyage</h1>
<p style="color: #888; margin-bottom: 30px; font-size: 1.1rem;">Upload a route file from the sidebar to visualize metrics.</p>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; max-width: 800px; margin: 0 auto;">
<div class="bento-card" style="padding: 30px;">
<div style="font-size: 2rem; margin-bottom: 10px;">🗺️</div>
<div style="font-weight: 600; color: white;">Interactive Maps</div>
</div>
<div class="bento-card" style="padding: 30px;">
<div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
<div style="font-weight: 600; color: white;">Fuel Analytics</div>
</div>
<div class="bento-card" style="padding: 30px;">
<div style="font-size: 2rem; margin-bottom: 10px;">🛥️</div>
<div style="font-weight: 600; color: white;">Route Optimization</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

