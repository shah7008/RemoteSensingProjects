"""
folium_visualization.py  –  Interactive map visualization of the Cold Chain
                              Temperature Prediction dataset using Folium
===========================================================================
Layers included:
  1. Route polyline       – vehicle/shipment GPS track
  2. Internal Temp markers– colour-coded circles (green/yellow/red by temp)
  3. Land Surface Temp    – HeatMap layer
  4. NDVI choropleth      – circle markers sized + coloured by NDVI
  5. Satellite Ambient Temp clusters – MarkerCluster layer

Run:
    python folium_visualization.py

Output:  cold_chain_map.html  (open in any browser)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import webbrowser

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Load dataset from CSV
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

csv_path = 'cold_chain_dataset.csv'
if not os.path.exists(csv_path):
    print(f"❌ ERROR: {csv_path} not found. Please run generate_dataset.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)

internal_temp       = df['internal_temperature_c'].values
ambient_temp        = df['ambient_temperature_c'].values
humidity            = df['humidity_percent'].values
door_open           = df['door_open'].values
vibration           = df['vibration_level'].values
latitude            = df['latitude'].values
longitude           = df['longitude'].values
satellite_amb_temp  = df['satellite_ambient_temp'].values
solar_radiation     = df['solar_radiation'].values
land_surface_temp   = df['land_surface_temp'].values
elevation           = df['elevation_m'].values
ndvi                = df['ndvi'].values
ndbi                = df['ndbi'].values

N_SAMPLES = len(df)
print(f"Dataset loaded from CSV  –  {N_SAMPLES} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Helper: colour internal temperature (target variable)
#     < 2 °C  → blue   (too cold)
#     2–6 °C  → green  (safe zone)
#     6–8 °C  → orange (warning)
#     > 8 °C  → red    (critical)
# ─────────────────────────────────────────────────────────────────────────────
def temp_color(temp: float) -> str:
    if temp < 2.0:
        return "#2196F3"   # blue
    elif temp <= 6.0:
        return "#4CAF50"   # green
    elif temp <= 8.0:
        return "#FF9800"   # orange
    else:
        return "#F44336"   # red

def ndvi_color(val: float) -> str:
    """Green gradient for NDVI (0 → grey, 1 → dark green)."""
    g = int(60 + val * 195)
    return f"#{g:02X}{min(g + 40, 255):02X}{g // 3:02X}"

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Subsample for performance (every 10th point → 200 markers)
# ─────────────────────────────────────────────────────────────────────────────
STEP = 10
idx  = np.arange(0, N_SAMPLES, STEP)

lats  = latitude[idx]
lons  = longitude[idx]
temps = internal_temp[idx]
lsts  = land_surface_temp[idx]
ndvis = ndvi[idx]
ndbi_vals = ndbi[idx]
sat_temps = satellite_amb_temp[idx]
hum_vals  = humidity[idx]
door_vals = door_open[idx]
vib_vals  = vibration[idx]
solar_vals= solar_radiation[idx]

center_lat = float(np.mean(lats))
center_lon = float(np.mean(lons))

print(f"Map centre  →  lat={center_lat:.4f}  lon={center_lon:.4f}")
print(f"Rendering {len(idx)} markers …")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Build the Folium map
# ─────────────────────────────────────────────────────────────────────────────
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="CartoDB dark_matter",
    prefer_canvas=True,
)

# ── Plugin controls ───────────────────────────────────────────────────────────
Fullscreen(position="topright", title="Fullscreen", title_cancel="Exit").add_to(m)
MiniMap(tile_layer="CartoDB dark_matter", toggle_display=True).add_to(m)

# ── A · GPS Route polyline ────────────────────────────────────────────────────
route_coords = list(zip(lats.tolist(), lons.tolist()))
folium.PolyLine(
    route_coords,
    color="#00BCD4",
    weight=2.5,
    opacity=0.6,
    tooltip="Cold-Chain Vehicle Route",
).add_to(folium.FeatureGroup(name="🚛 GPS Route", show=True).add_to(m))

# ── B · Internal Temperature markers (colour-coded) ──────────────────────────
temp_fg = folium.FeatureGroup(name="🌡️ Internal Temperature", show=True)
for i in range(len(idx)):
    lat, lon = float(lats[i]), float(lons[i])
    tmp  = float(temps[i])
    hum  = float(hum_vals[i])
    door = "Yes" if door_vals[i] > 0.5 else "No"
    vib  = float(vib_vals[i])
    solar= float(solar_vals[i])

    popup_html = f"""
    <div style='font-family:Arial;font-size:12px;width:220px;'>
      <b style='color:#FF6B6B;font-size:14px;'>📦 Cold Chain Reading #{i*STEP+1}</b><hr/>
      <b>Internal Temp:</b> {tmp:.2f} °C<br/>
      <b>Humidity:</b>      {hum:.1f} %<br/>
      <b>Door Open:</b>     {door}<br/>
      <b>Vibration:</b>     {vib:.3f}<br/>
      <b>Solar Radiation:</b> {solar:.1f} W/m²<br/>
      <b>Coords:</b>        ({lat:.4f}, {lon:.4f})
    </div>"""

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=temp_color(tmp),
        fill=True,
        fill_color=temp_color(tmp),
        fill_opacity=0.85,
        tooltip=f"Internal Temp: {tmp:.2f} °C",
        popup=folium.Popup(popup_html, max_width=240),
    ).add_to(temp_fg)
temp_fg.add_to(m)

# ── C · Land Surface Temperature HeatMap ─────────────────────────────────────
lst_fg  = folium.FeatureGroup(name="🔥 Land Surface Temp (HeatMap)", show=True)
lst_min = float(lsts.min())
lst_max = float(lsts.max())
heat_data = [
    [float(lats[i]), float(lons[i]), float((lsts[i] - lst_min) / (lst_max - lst_min + 1e-9))]
    for i in range(len(idx))
]
HeatMap(
    heat_data,
    min_opacity=0.3,
    max_zoom=18,
    radius=18,
    blur=12,
    gradient={"0.2": "#2196F3", "0.5": "#FFEB3B", "0.8": "#FF9800", "1.0": "#F44336"},
).add_to(lst_fg)
lst_fg.add_to(m)

# ── D · NDVI circle markers (sized by NDVI, coloured green gradient) ──────────
ndvi_fg = folium.FeatureGroup(name="🌿 NDVI (Vegetation Index)", show=False)
for i in range(len(idx)):
    lat, lon  = float(lats[i]), float(lons[i])
    ndvi_val  = float(ndvis[i])
    ndbi_val  = float(ndbi_vals[i])
    radius    = 4 + ndvi_val * 12   # 4–16 px

    popup_html = f"""
    <div style='font-family:Arial;font-size:12px;width:180px;'>
      <b style='color:#4CAF50;font-size:14px;'>🌿 Remote Sensing</b><hr/>
      <b>NDVI:</b>  {ndvi_val:.4f}<br/>
      <b>NDBI:</b>  {ndbi_val:.4f}<br/>
      <b>LST:</b>   {float(lsts[i]):.2f} °C<br/>
      <b>Coords:</b>({lat:.4f}, {lon:.4f})
    </div>"""

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=ndvi_color(ndvi_val),
        fill=True,
        fill_color=ndvi_color(ndvi_val),
        fill_opacity=0.75,
        tooltip=f"NDVI: {ndvi_val:.4f}",
        popup=folium.Popup(popup_html, max_width=200),
    ).add_to(ndvi_fg)
ndvi_fg.add_to(m)

# ── E · Satellite Ambient Temp – MarkerCluster ────────────────────────────────
sat_fg      = folium.FeatureGroup(name="🛰️ Satellite Ambient Temp (Cluster)", show=False)
sat_cluster = MarkerCluster(name="Satellite Temp Clusters").add_to(sat_fg)
for i in range(len(idx)):
    lat, lon = float(lats[i]), float(lons[i])
    sat_t    = float(sat_temps[i])

    folium.Marker(
        location=[lat, lon],
        tooltip=f"Sat. Ambient: {sat_t:.2f} °C",
        popup=folium.Popup(
            f"<b>Satellite Ambient Temp:</b> {sat_t:.2f} °C<br/>"
            f"<b>Solar Radiation:</b> {float(solar_vals[i]):.1f} W/m²",
            max_width=200,
        ),
        icon=folium.Icon(
            color="blue" if sat_t < 22 else ("orange" if sat_t < 24 else "red"),
            icon="cloud",
            prefix="fa",
        ),
    ).add_to(sat_cluster)
sat_fg.add_to(m)

# ── F · Legend HTML ───────────────────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed; bottom: 40px; left: 40px; z-index: 9999;
    background: rgba(10,10,20,0.88);
    border: 1px solid #444; border-radius: 10px;
    padding: 14px 18px; font-family: Arial; font-size: 13px; color: #eee;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
  <b style="font-size:15px; color:#00BCD4;">🌡️ Cold Chain Map – Legend</b><br/><br/>
  <b>Internal Temperature</b><br/>
  <span style="color:#2196F3;">●</span> &lt; 2 °C &nbsp; Too Cold<br/>
  <span style="color:#4CAF50;">●</span> 2–6 °C &nbsp; Safe Zone<br/>
  <span style="color:#FF9800;">●</span> 6–8 °C &nbsp; Warning<br/>
  <span style="color:#F44336;">●</span> &gt; 8 °C &nbsp; Critical<br/><br/>
  <b>Other Layers</b><br/>
  <span style="color:#00BCD4;">─</span> GPS Route<br/>
  🔥 LST HeatMap (blue→red)<br/>
  🌿 NDVI (circle size = NDVI)<br/>
  🛰️ Satellite Temp (clusters)
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── G · Layer Control ─────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False, position="topright").add_to(m)

# ── H · Title banner ─────────────────────────────────────────────────────────
title_html = """
<div style="
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 9999; background: rgba(10,10,20,0.85);
    border: 1px solid #00BCD4; border-radius: 8px;
    padding: 8px 20px; font-family: Arial; font-size: 15px; color: #00BCD4;
    box-shadow: 0 2px 10px rgba(0,188,212,0.3);">
  🚛 Cold Chain Temperature Prediction – Remote Sensing Dataset Visualisation
</div>"""
m.get_root().html.add_child(folium.Element(title_html))

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Save and open
# ─────────────────────────────────────────────────────────────────────────────
output_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "cold_chain_map.html")
m.save(output_path)

print(f"\n✅  Map saved  →  {output_path}")
print("   Opening in browser …")
webbrowser.open(f"file:///{output_path.replace(os.sep, '/')}")
