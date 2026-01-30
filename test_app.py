import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from urllib.request import urlopen, Request

# Try importing Pillow, handle if missing
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
from skyfield.framelib import itrs
from datetime import timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Satellite Tracker Pro")
st.title("🛰️ Satellite Tracker Pro: Orbit, Visibility & Proximity")

# --- Constants ---
R_EARTH = 6371 # Earth radius in km

# Backup TLE data
FALLBACK_TLE = """
ISS (ZARYA)
1 25544U 98067A   24017.50000000  .00016717  00000+0  30693-3 0  9993
2 25544  51.6416 280.6231 0005697 325.7688 154.5427 15.49678367491472
HST
1 20580U 90037B   24017.50000000  .00001500  00000+0  10000-3 0  9991
2 20580  28.4699 265.1234 0002500 100.0000 260.0000 15.09000000  1000
GPS BIIA-10 (PRN 32)
1 20959U 90103A   24017.50000000 -.00000050  00000+0  00000+0 0  9995
2 20959  54.8500 100.0000 0150000  45.0000 315.0000  2.00565432 10000
STARLINK-1007
1 44713U 19074A   24017.50000000  .00000678  00000+0  67856-4 0  9995
2 44713  53.0543 178.9876 0001423  87.9752 272.1462 15.06394628230052
"""

# --- Helper Functions ---

def get_category(name):
    """Categorizes satellites based on their name."""
    name = name.upper()
    if 'STARLINK' in name: return 'Starlink'
    elif 'ONEWEB' in name: return 'OneWeb'
    elif 'GPS' in name: return 'GPS'
    elif 'BEIDOU' in name: return 'Beidou'
    elif 'GALILEO' in name: return 'Galileo'
    elif 'GLONASS' in name: return 'GLONASS'
    elif 'IRIDIUM' in name: return 'Iridium'
    elif 'ISS' in name: return 'ISS'
    else: return 'Other'

def get_footprint(lat, lon, alt):
    """Calculate the 3D Cartesian coordinates of the satellite's visibility footprint."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    alpha = np.arccos(R_EARTH / (R_EARTH + alt))
    theta = np.linspace(0, 2 * np.pi, 100)

    lat_circle = np.arcsin(
        np.sin(lat_rad) * np.cos(alpha) +
        np.cos(lat_rad) * np.sin(alpha) * np.cos(theta)
    )

    lon_circle = lon_rad + np.arctan2(
        np.sin(theta) * np.sin(alpha) * np.cos(lat_rad),
        np.cos(alpha) - np.sin(lat_rad) * np.sin(lat_circle)
    )

    R_plot = R_EARTH + 10
    x = R_plot * np.cos(lat_circle) * np.cos(lon_circle)
    y = R_plot * np.cos(lat_circle) * np.sin(lon_circle)
    z = R_plot * np.sin(lat_circle)
    return x, y, z

def get_trajectory(sat, ts, t_start, duration_minutes=90):
    """Propagates the orbit forward to generate a trajectory line."""
    start_dt = t_start.utc_datetime()
    time_list = [start_dt + timedelta(minutes=i) for i in range(duration_minutes)]
    times = ts.from_datetimes(time_list)

    geocentric = sat.at(times)
    subpoint = wgs84.subpoint(geocentric)

    lats = subpoint.latitude.radians
    lons = subpoint.longitude.radians
    alts = subpoint.elevation.km

    rs = R_EARTH + alts
    xs = rs * np.cos(lats) * np.cos(lons)
    ys = rs * np.cos(lats) * np.sin(lons)
    zs = rs * np.sin(lats)

    return xs, ys, zs

def check_visibility(sat, t, ground_lat, ground_lon):
    """Checks if satellite is visible from a ground station."""
    ground_station = wgs84.latlon(ground_lat, ground_lon)
    difference = sat - ground_station
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > 0, alt.degrees, distance.km

# --- Caching Functions ---

@st.cache_resource
def load_satellites():
    """
    Loads satellite data with a priority system:
    1. Local 'active.txt' file (Fastest, works in Cloud)
    2. CelesTrak URL (Live, requires internet)
    3. Fallback String (Emergency only)
    """
    lines = []
    source = "Unknown"

    # 1. Try Local File (Pre-cached)
    if os.path.exists("active.txt"):
        try:
            with open("active.txt", "rb") as f:
                lines = [line for line in f]
            source = "Local File"
        except Exception as e:
            print(f"Local file read error: {e}")

    # 2. Try Live Download (if no local file)
    if not lines:
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=30) as response:
                lines = [line for line in response.readlines()]
            source = "CelesTrak (Live)"
        except Exception as e:
            st.warning(f"⚠️ Could not connect to CelesTrak: {e}")

    # 3. Use Fallback
    if not lines:
        source = "Emergency Fallback"
        lines = [line.encode('ascii') for line in FALLBACK_TLE.strip().splitlines()]

    # Final Validation
    if not lines or len(lines) < 3:
        st.error("❌ No valid TLE data found.")
        return [], "None"

    # Ensure binary format for Skyfield
    if isinstance(lines[0], str):
        lines = [l.encode('ascii') for l in lines]

    ts = load.timescale(builtin=True)
    satellites = list(parse_tle_file(lines, ts))
    return satellites, source

@st.cache_data
def get_geometry():
    """Generates Earth mesh and fetches landmass data."""
    # 1. Sphere Geometry (Royal Blue)
    N = 100
    phi = np.linspace(0, 2 * np.pi, N)
    theta = np.linspace(0, np.pi, N)
    phi, theta = np.meshgrid(phi, theta)
    x_earth = R_EARTH * np.sin(theta) * np.cos(phi)
    y_earth = R_EARTH * np.sin(theta) * np.sin(phi)
    z_earth = R_EARTH * np.cos(theta)

    # 2. Landmasses (Green Polygons)
    land_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"
    land_xc, land_yc, land_zc = [], [], []
    try:
        with urlopen(land_url, timeout=5) as response:
            land_json = json.load(response)

        for feature in land_json['features']:
            geom_type = feature['geometry']['type']
            coordinates = feature['geometry']['coordinates']

            # Helper to process a ring (exterior or interior)
            def process_ring(ring_coords):
                lons, lats = zip(*ring_coords)
                clat, clon = np.radians(np.array(lats)), np.radians(np.array(lons))
                R_land = R_EARTH + 10 # Slightly above Earth to be visible
                land_xc.extend(R_land * np.cos(clat) * np.cos(clon))
                land_yc.extend(R_land * np.cos(clat) * np.sin(clon))
                land_zc.extend(R_land * np.sin(clat))
                land_xc.append(None); land_yc.append(None); land_zc.append(None) # Disconnect

            if geom_type == 'Polygon':
                # Exterior ring
                process_ring(coordinates[0])
                # Interior rings (holes)
                for i_ring in coordinates[1:]:
                    process_ring(i_ring)
            elif geom_type == 'MultiPolygon':
                for polygon_coords in coordinates:
                    # Exterior ring of each polygon
                    process_ring(polygon_coords[0])
                    # Interior rings of each polygon\n                    for i_ring in polygon_coords[1:]:\n                        process_ring(i_ring)\n\n    except Exception: # Catch any errors during GeoJSON download/processing\n        land_xc, land_yc, land_zc = [], [], [] # Ensure empty lists on failure\n\n    return (x_earth, y_earth, z_earth), (land_xc, land_yc, land_zc)\n\n@st.cache_data\ndef process_positions(_satellites):\n    \"\"\"Calculates current positions for all satellites.\"\"\"\n    ts = load.timescale(builtin=True)\n    t = ts.now()\n    data = []\n    for i, sat in enumerate(_satellites):\n        try:\n            geocentric = sat.at(t)\n            subpoint = wgs84.subpoint(geocentric)\n            lat, lon, alt = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km\n\n            r = R_EARTH + alt\n            rad_lat, rad_lon = np.radians(lat), np.radians(lon)\n            x = r * np.cos(rad_lat) * np.cos(rad_lon)\n            y = r * np.cos(rad_lat) * np.sin(rad_lon)\n            z = r * np.sin(rad_lat)\n\n            cat = get_category(sat.name)\n\n            data.append({'Name': sat.name, 'Category': cat, 'Lat': lat, 'Lon': lon, 'Alt': alt, 'X': x, 'Y': y, 'Z': z, 'Index': i})\n        except Exception: continue\n    return pd.DataFrame(data), t\n\n# --- Main Execution ---\n\nwith st.spinner(\"Initializing Satellite Data...\"):\n    satellites, source = load_satellites()\n\n    if not satellites: st.stop()\n\n    # Unpack updated geometry tuple\n    (x_earth, y_earth, z_earth), (land_xc, land_yc, land_zc) = get_geometry()\n    df, t_now = process_positions(satellites)\n\nif df.empty: st.stop()\n\n# --- Sidebar Controls ---\n\nst.sidebar.header(\"Target Selection\")\nst.sidebar.info(f\"Data Source: {source} ({len(df)} sats)\")\n\nif 'selected_sat_name' not in st.session_state:\n    sl_match = df[df['Name'].str.contains('STARLINK')]\n    st.session_state.selected_sat_name = sl_match.iloc[0]['Name'] if not sl_match.empty else df.iloc[0]['Name']\n\ndef update_selection(): pass\n\nsearch = st.sidebar.text_input(\"Search Satellite\", placeholder=\"ISS\").upper()\noptions = df[df['Name'].str.contains(search, case=False)]['Name'].sort_values() if search else df['Name'].sort_values()\n\nif st.session_state.selected_sat_name not in options.values:\n    if len(options) > 0: st.session_state.selected_sat_name = options.iloc[0]\n\nselected_name = st.sidebar.selectbox(\"Select Satellite\", options, key='selected_sat_name', on_change=update_selection)\n\nst.sidebar.markdown(\"---\")\nst.sidebar.header(\"Visualization Options\")\nshow_traj = st.sidebar.checkbox(\"Show 90min Orbit Path\", value=True)\n\nst.sidebar.markdown(\"---\")\nst.sidebar.header(\"Ground Station\")\ngs_lat = st.sidebar.number_input(\"Latitude\", value=40.7128, min_value=-90.0, max_value=90.0)\ngs_lon = st.sidebar.number_input(\"Longitude\", value=-74.0060, min_value=-180.0, max_value=180.0)\n\n# --- Main Logic ---\n\nif selected_name:\n    row = df[df['Name'] == selected_name].iloc[0]\n    sat_idx = int(row['Index'])\n    sat_obj = satellites[sat_idx]\n\n    fx, fy, fz = get_footprint(row['Lat'], row['Lon'], row['Alt'])\n    tx, ty, tz = [], [], []\n    if show_traj:\n        ts = load.timescale(builtin=True)\n        tx, ty, tz = get_trajectory(sat_obj, ts, t_now)\n\n    is_visible, el_deg, dist_km = check_visibility(sat_obj, t_now, gs_lat, gs_lon)\n\n    dists = np.sqrt((df['X'] - row['X'])**2 + (df['Y'] - row['Y'])**2 + (df['Z'] - row['Z'])**2)\n    dists[dists == 0] = np.inf\n    min_dist_idx = dists.idxmin()\n    nearest_sat = df.loc[min_dist_idx]\n    min_dist_km = dists[min_dist_idx]\n\n    gs_rad_lat, gs_rad_lon = np.radians(gs_lat), np.radians(gs_lon)\n    gs_x = R_EARTH * np.cos(gs_rad_lat) * np.cos(gs_rad_lon)\n    gs_y = R_EARTH * np.cos(gs_rad_lat) * np.sin(gs_rad_lon)\n    gs_z = R_EARTH * np.sin(gs_rad_lat)\n\n    # --- Visualization ---\n    fig = go.Figure()\n\n    # Earth Surface (Solid Royal Blue)\n    fig.add_trace(go.Surface(\n        x=x_earth, y=y_earth, z=z_earth,\n        colorscale=[[0, 'royalblue'], [1, 'royalblue']],\n        showscale=False, opacity=1.0, hoverinfo='skip', name='Earth'\n    ))\n\n    # Landmasses (Solid Green) - now uses land_xc, land_yc, land_zc\n    if len(land_xc) > 0:\n        fig.add_trace(go.Scatter3d(x=land_xc, y=land_yc, z=land_zc,\n            mode='lines', line=dict(color='green', width=1), hoverinfo='skip', name='Land'))\n\n    colors = {'Starlink': '#32CD32', 'OneWeb': '#FFD700', 'GPS': '#FF4500', 'Beidou': '#FF1493',\n              'Galileo': '#00FFFF', 'GLONASS': '#FFA500', 'Iridium': '#1E90FF', 'ISS': '#FFFFFF', 'Other': '#808080'}\n    for cat, color in colors.items():\n        df_cat = df[df['Category'] == cat]\n        if not df_cat.empty:\n            fig.add_trace(go.Scatter3d(x=df_cat['X'], y=df_cat['Y'], z=df_cat['Z'], mode='markers',\n                marker=dict(size=4, color=color, opacity=0.0),\n                name=cat, text=df_cat['Name'],\n                customdata=df_cat['Name'].tolist(), hoverinfo='text'))\n\n    if show_traj:\n        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines', line=dict(color='orange', width=4, dash='dot'), name='Orbit Path', hoverinfo='skip'))\n    fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='yellow', width=5), name='Footprint', hoverinfo='skip'))\n    fig.add_trace(go.Scatter3d(x=[row['X']], y=[row['Y']], z=[row['Z']], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name=selected_name, hoverinfo='text', text=selected_name))\n\n    gs_col = 'green' if is_visible else 'gray'\n    fig.add_trace(go.Scatter3d(x=[gs_x], y=[gs_y], z=[gs_z], mode='markers', marker=dict(size=8, color=gs_col, symbol='x'), name='Ground Station', hoverinfo='text', text='Ground Station'))\n    if is_visible:\n        fig.add_trace(go.Scatter3d(x=[gs_x, row['X']], y=[gs_y, row['X']], z=[gs_z, row['Z']], mode='lines', line=dict(color='green', width=3), name='Line of Sight', hoverinfo='skip'))\n\n    fig.add_trace(go.Scatter3d(x=[row['X'], nearest_sat['X']], y=[row['Y'], nearest_sat['Y']], z=[row['Z'], nearest_sat['Z']],\n                               mode='lines', line=dict(color='magenta', width=2), name=f\"Nearest: {nearest_sat['Name']}\", hoverinfo='skip'))\n\n    fig.update_layout(\n        template='plotly_dark',\n        dragmode='orbit',\n        uirevision='constant',\n        clickmode='event+select',\n        scene=dict(\n            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),\n            xaxis=dict(visible=False, range=[-50000, 50000]),\n            yaxis=dict(visible=False, range=[-50000, 50000]),\n            zaxis=dict(visible=False, range=[-50000, 50000]),\n            camera=dict(projection=dict(type='perspective'), center=dict(x=0, y=0, z=0), eye=dict(x=1.2, y=1.2, z=1.2))\n        ),\n        margin=dict(r=0, t=0, l=0, b=0),\n        height=800,\n        modebar=dict(remove=['pan3d', 'select2d', 'lasso2d'])\n    )\n\n    m1, m2, m3 = st.columns(3)\n    m1.metric(\"Altitude\", f\"{row['Alt']:.1f} km\", f\"Category: {row['Category']}\")\n    m2.metric(\"Station Visibility\", \"VISIBLE\" if is_visible else \"No Signal\", f\"{el_deg:.1f}\u00b0 El\")\n    m3.metric(\"Nearest Neighbor\", nearest_sat['Name'], f\"{min_dist_km:.1f} km away\")\n\n    event = st.plotly_chart(fig, width=\"stretch\", on_select=\"rerun\", selection_mode=\"points\", key=\"main_map\")\n\n    if event and event.selection and event.selection.points:\n        point = event.selection.points[0]\n        if 'customdata' in point:\n            clicked = point['customdata']\n            if clicked != st.session_state.selected_sat_name:\n                st.session_state.selected_sat_name = clicked\n                st.rerun()\n\n    with st.expander(\"Debug: Raw Event Data\"):\n        st.write(event)"
      },
      "cellId": "555e3e01",
      "cellMetadata": {
        "lastEditedTime": "2026-01-30T09:00:03Z"
      }
    }
  }
}
