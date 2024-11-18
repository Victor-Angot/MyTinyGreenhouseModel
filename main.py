import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from math import *

L = 50.0
l = 9.0
H = 3.0

DEFAULT_PARAMS = {
    'U': 0.76,
    'sigma': 5.67e-8,
    'A_cover': 2*(L*H + l*H) + L*l,
    'h': 1.05,
    'tau_cover': 0.83,
    'V_zone': L*l*H, #m3
    'A_floor': L*l,
    'Wsat': 0.033, #saturaed moisture kg/m3
    'Wi': 0.01724, #average moisture kg/m3
    'I_global': 780.0,
    'epsilon_i': 0.92,
    'epsilon_floor': 0.95,
    'epsilon_sky': 0.8,
    'LAI': 3.0,
    'rho_air': 1.2,
    'c_air': 1008.0,
    'r_b': 275.0,
    'cp': 4185, #2.45e6,
    'm_target_star': 0.7,
    'delta_H' : 2.45e6,
    'delta_t' : 3600.0,
    'f_sens' : 0.1, # fraction maximal sensible heat TODO revoir ce que c'est
    'f_lat' : 0.9, # fraction chaleur latente maximal TODO revoir ce que c'est
    'T_target' : 12.0,
    'T_i': 28.0,          
    'T_0': 31.0,          
    'T_cover': 40.0,      
    'T_sky': 10.0,
    'T_floor': 35.0,
    'hour': "12:00",      
}

# Initialize session_state with default values
for key, value in DEFAULT_PARAMS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset parameters
def reset_parameters():
    for key, value in DEFAULT_PARAMS.items():
        st.session_state[key] = value

# Thermal bilan
def Q_cond(T_cover, T_0, A_cover):
    '''Q_cond represents the conduction heat transfer rate between the outside air and the covering materials'''
    return DEFAULT_PARAMS['U']*A_cover*(T_cover-T_0)

def Q_conv(T_cover, T_0, A_cover):
    '''Q_conv represents the convective heat transfer rate between the outside air and the covering material'''
    return DEFAULT_PARAMS['h']*A_cover*(T_cover-T_0)

def Q_solar(I_global, A_floor):
    '''Q_solar represents the solar heat transfer rate due to direct and diffused shortwave solar 
    radiation transmitted through the covering materials'''
    return DEFAULT_PARAMS['tau_cover']*A_floor*I_global 

def Q_long_sky(T_sky, A_cover):
    '''Q_long_sky represents the long-wavelength radiation heat transfer rate between the greenhouse and the sky'''
    return DEFAULT_PARAMS['sigma']*A_cover*DEFAULT_PARAMS['tau_cover']*(DEFAULT_PARAMS['epsilon_sky']*(T_sky+273.15)**4)

def Q_long_cover(T_i, A_cover):
    '''Q_long_cover represents the long-wavelength radiation heat transfer rate between the greenhouse and the cover'''
    return DEFAULT_PARAMS['sigma']*A_cover*DEFAULT_PARAMS['tau_cover']*(DEFAULT_PARAMS['epsilon_i']*(T_i+273.15)**4)

# def Q_long(T_i, T_sky, A_cover):
#     '''Q_long represents the long-wavelength radiation heat transfer rate of
#     the greenhouse through the covering materials'''
#     return DEFAULT_PARAMS['sigma']*A_cover*DEFAULT_PARAMS['tau_cover']*(DEFAULT_PARAMS['epsilon_i']*(T_i+273.15)**4 - DEFAULT_PARAMS['epsilon_sky']*(T_sky+273.15)**4)

def Q_crop(T_i, A_floor):
    '''Q_crop is due to the difference between plant leaf temperature and greenhouse indoor air temperature''' 
    T_plant = T_i - 2
    return 2*A_floor*DEFAULT_PARAMS['LAI']*(DEFAULT_PARAMS['rho_air']*DEFAULT_PARAMS['c_air']*(T_i-T_plant))/DEFAULT_PARAMS['r_b']

def calcul_m_evap (Wsat, Wi, rho_air, V_zone, m_target_star, f_lat):
    m_evap_zonemax = (Wsat - Wi) * rho_air * V_zone
    m_evap_flowmax = m_target_star
    #m_evap_max = min(m_evap_zonemax , m_evap_flowmax)
    m_evap = f_lat * m_evap_zonemax
    return m_evap

def Q_lat(delta_t):
    '''Q_lat is the maximum latent heat exchange rate for the zone and fog droplets'''
    m_evap = calcul_m_evap(DEFAULT_PARAMS['Wsat'], DEFAULT_PARAMS['Wi'], DEFAULT_PARAMS['rho_air'], DEFAULT_PARAMS['V_zone'], DEFAULT_PARAMS['m_target_star'], DEFAULT_PARAMS['f_lat'])
    return DEFAULT_PARAMS['delta_H']*m_evap/delta_t

def Q_sens(T_i, T_target):
    '''Q_sens is the sensible heat exchange rate of the zone and fog droplets'''
    return DEFAULT_PARAMS['f_sens']*DEFAULT_PARAMS['cp']*DEFAULT_PARAMS['m_target_star']*(T_i-T_target)

def Q_fog(T_i, T_target, delta_t):
    Q_lat_val = Q_lat(delta_t)
    Q_sens_val = Q_sens(T_i, T_target)
    return Q_sens_val + Q_lat_val

def Q_floor_rad(T_floor):
    k = 0.5
    f_sol = exp(-k*DEFAULT_PARAMS['LAI']) # fraction of soil not covered by plants
    return DEFAULT_PARAMS['epsilon_floor']*DEFAULT_PARAMS['sigma']*DEFAULT_PARAMS['A_floor']*f_sol*(T_floor+273.15)**4

if 'reset_button_clicked' not in st.session_state:
    st.session_state['reset_button_clicked'] = False

if st.session_state['reset_button_clicked']:
    reset_parameters()
    st.session_state['reset_button_clicked'] = False

st.title("My Tiny Greenhouse Model")
st.sidebar.header("Greenhouse parameters")
st.sidebar.number_input("Cover Surface Area (A_cover) [m²]", min_value=1.0, key='A_cover')
st.sidebar.number_input("Floor Surface Area (A_floor) [m²]", min_value=1.0, key='A_floor')
st.sidebar.number_input("Misting period (delta_t) [s]", min_value=1.0, key='delta_t')

st.sidebar.header("Greenhouse overview") # st.image if not wanted in the sidebar
image_path = "./greenhouse.png"  
st.sidebar.image(image_path, caption='Heat flows in the greenhouse', use_column_width=True)

# Initialize session state variables
if 'time_str' not in st.session_state:
    st.session_state.time_str = "12:00"

if 'prev_time_str' not in st.session_state:
    st.session_state.prev_time_str = st.session_state.time_str

# Create a list of times in 30-minute intervals
times = [datetime(2021, 1, 1, hour=h, minute=m).strftime("%H:%M")
         for h in range(0, 24)
         for m in (0, 30)]

# Time selection slider using st.select_slider
time_str = st.select_slider("Select the time of day", options=times, value=st.session_state.hour)
st.session_state.time_str = time_str

# Check if time has changed
if st.session_state.time_str != st.session_state.prev_time_str:
    # Time has changed, recalculate temperatures

    # Convert time string to decimal hour
    h, m = map(int, st.session_state.time_str.split(':'))
    hour_mod = h + m / 60.0
    if hour_mod == 24.0:
        hour_mod = 0.0  # Wrap around to 0

    times = [
        0.0, 0.5, 1.0, 1.5, 2.0,
        2.5, 3.0, 3.5, 4.0, 4.5,
        5.0, 5.5, 6.0, 6.5, 7.0,
        7.5, 8.0, 8.5, 9.0, 9.5,
        10.0, 10.5, 11.0, 11.5, 12.0,
        12.5, 13.0, 13.5, 14.0, 14.5,
        15.0, 15.5, 16.0, 16.5, 17.0,
        17.5, 18.0, 18.5, 19.0, 19.5,
        20.0, 20.5, 21.0, 21.5, 22.0,
        22.5, 23.0, 23.5, 24.0
    ]

    temperatures = [
        30.5, 29.8, 29.0, 28.6, 28.2,
        26.1, 25.8, 25.3, 24.5, 24.3,
        24.3, 24.3, 24.1, 23.8, 23.7,
        23.4, 24.4, 25.6, 26.9, 27.8,
        28.9, 29.7, 30.2, 31.9, 32.4,
        32.5, 33.8, 34.7, 35.7, 36.2,
        37.0, 37.4, 37.2, 38.0, 38.3,
        38.3, 38.5, 38.2, 37.7, 37.2,
        36.2, 35.4, 34.4, 33.6, 32.8,
        32.1, 31.3, 30.8, 30.5
    ]

    # Linear interpolation of outdoor temperature (T_0)
    T_0 = np.interp(hour_mod, times, temperatures)

    st.session_state.T_0 = T_0
    st.session_state.T_i = T_0 + 2       # Indoor temperature
    st.session_state.T_cover = T_0 + 5   # Outdoor wall temperature

    # Calculation of global solar radiation (I_global)
    # Assume sunrise at 6h and sunset at 18h
    sunrise = 6.0
    sunset = 18.0
    if sunrise <= hour_mod <= sunset:
        # Calculate solar angle
        solar_angle = np.pi * (hour_mod - sunrise) / (sunset - sunrise)
        I_global = 1000 * np.sin(solar_angle)  # Maximum intensity of 1000 W/m²
        if I_global < 0:
            I_global = 0
    else:
        I_global = 0  # No solar radiation at night

    st.session_state.I_global = I_global

    st.session_state.prev_time_str = st.session_state.time_str

st.sidebar.header("Temperature and solar radiation")
st.sidebar.slider("Global solar radiation [W/m²]", 0.0, 1360.0, key='I_global')
st.sidebar.slider("Indoor temperature (T_i) [°C]", -10.0, 50.0, key='T_i')
st.sidebar.slider("Outdoor temperature (T_0) [°C]", -10.0, 50.0, key='T_0')
st.sidebar.slider("Outdoor wall temperature [°C]", -10.0, 50.0, key='T_cover')
st.sidebar.slider("Sky temperature [°C]", -10.0, 20.0, key='T_sky')
st.sidebar.slider("Floor temperature [°C]", -10.0, 50.0, key='T_floor')
st.sidebar.slider("Fog water temperature [°C]", 0.0, 40.0, key='T_target')

if st.sidebar.button("🔄 Reset to default"):
    st.session_state['reset_button_clicked'] = True
    st.rerun()

T_i = st.session_state['T_i']
T_0 = st.session_state['T_0']
T_cover = st.session_state['T_cover']
T_sky = st.session_state['T_sky']
T_floor = st.session_state['T_floor']
T_target = st.session_state['T_target']
A_cover = st.session_state['A_cover']
A_floor = st.session_state['A_floor']
I_global = st.session_state['I_global']
delta_t = st.session_state['delta_t']
# Calculate heat transfer rates
Q_cond_val = Q_cond(T_cover, T_0, A_cover)
Q_conv_val = Q_conv(T_cover, T_0, A_cover)
Q_cond_conv_val = abs(Q_cond_val + Q_conv_val)
Q_solar_val = Q_solar(I_global, A_floor)
Q_floor_rad_val = Q_floor_rad(T_floor)
Q_sky_val = Q_long_sky(T_sky, A_cover)
Q_cover_val = Q_long_cover(T_i, A_cover)
Q_long_val = Q_sky_val + Q_cover_val + Q_floor_rad_val
Q_crop_val = Q_crop(T_i, A_floor)
Q_lat_val = Q_lat(delta_t)
Q_sens_val = Q_sens(T_i, T_target) 
Q_fog_val = Q_fog(T_i, T_target, delta_t)
net_heat = Q_cond_conv_val + Q_solar_val + Q_long_val + Q_crop_val + Q_fog_val + Q_floor_rad_val

# Prepare data for the pie chart
flows = {
    'Conduction & Convection': Q_cond_conv_val,
    'Solar Radiation': Q_solar_val,
    'IR Radiation (Floor, Sky, Cover)': Q_long_val,
    'Heat Exchange with Crops': Q_crop_val,
    'Heat Exchange with Fog': Q_fog_val,
}

# Initialize the 'Others' sum
others_value = 0

# Create a new dictionary for the flows to be plotted
flows_pie = {}

# Iterate over the flows and process them
for flow_name, flow_value in flows.items():
    if flow_value > 0:
        if flow_value < 5000:
            # Add to 'Others' sum
            others_value += flow_value
        else:
            # Include in the flows to be plotted
            flows_pie[flow_name] = flow_value

# If there is any value in 'Others', add it to the flows_pie dictionary
if others_value > 0:
    flows_pie['Others'] = others_value

# Create the pie chart
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(
    flows_pie.values(),
    labels=flows_pie.keys(),
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab20.colors
)
ax_pie.axis('equal')  # Ensure the pie chart is circular
st.subheader("Distribution of Heat Flows")
st.pyplot(fig_pie)

# Prepare data for the Sankey diagram
labels_sankey = [
    'Solar Radiation',
    'Sky Radiation',
    'Cover Radiation',
    'Floor Radiation',
    'Conduction',
    'Convection',
    'Heat Exchange with Crops',
    'Heat Exchange with Fog',
    'Cold sources',
    'Hot sources',
    'Net Heaflow'
]

# Indices des labels
solar_idx = labels_sankey.index('Solar Radiation')
sky_idx = labels_sankey.index('Sky Radiation')
conduction_idx = labels_sankey.index('Conduction')
convection_idx = labels_sankey.index('Convection')
cover_idx = labels_sankey.index('Cover Radiation')
floor_idx = labels_sankey.index('Floor Radiation')
crop_idx = labels_sankey.index('Heat Exchange with Crops')
fog_idx = labels_sankey.index('Heat Exchange with Fog')
cold_idx = labels_sankey.index('Cold sources')
hot_idx = labels_sankey.index('Hot sources')
net_idx = labels_sankey.index('Net Heaflow')

source = [
    solar_idx, sky_idx, conduction_idx, convection_idx,
    floor_idx, crop_idx, fog_idx,
    cover_idx, cover_idx,  # "Cover Radiation" divisé en deux
    cold_idx, hot_idx
]

target = [
    hot_idx, hot_idx, cold_idx, cold_idx,
    hot_idx, cold_idx, cold_idx,
    cold_idx, hot_idx,  # Flux divisés de "Cover Radiation"
    net_idx, net_idx
]

value = [
    Q_solar_val,          # Solar Radiation -> Hot sources
    Q_sky_val,            # Sky Radiation -> Hot sources
    Q_cond_val,           # Conduction -> Cold sources
    Q_conv_val,           # Convection -> Cold sources
    Q_floor_rad_val,      # Floor Radiation -> Cold sources
    Q_crop_val,           # Heat Exchange with Crops -> Cold sources
    Q_fog_val,            # Heat Exchange with Fog -> Cold sources
    Q_cover_val / 2,      # Cover Radiation -> Cold sources
    Q_cover_val / 2,      # Cover Radiation -> Hot sources
    None,                 # Cold sources -> Net Heaflow (à calculer)
    None                  # Hot sources -> Net Heaflow (à calculer)
]

cold_flow_total = Q_cond_val + Q_conv_val + Q_crop_val + Q_fog_val + (Q_cover_val / 2)
hot_flow_total = Q_solar_val + Q_sky_val + (Q_cover_val / 2) + Q_floor_rad_val

value[-2] = cold_flow_total  # Cold sources -> Net Heaflow
value[-1] = hot_flow_total   # Hot sources -> Net Heaflow

node_colors = [
    '#FFD700',   # Solar Radiation
    '#D2FFFF',   # Sky Radiation
    '#b57cff',   # Cover Radiation
    '#5D1600',   # Convection
    '#FF4500',   # Conduction
    '#00CED1',   # Floor Radiation
    '#32CD32',   # Heat Exchange with Crops
    '#7CFF7E',   # Heat Exchange with Fog
    '#0061ff',   # Cold sources
    '#e20f0f',   # Hot sources
    '#808080',   # Net Heaflow
]

# Mise à jour des couleurs des flux en utilisant la couleur du nœud source
link_colors = [
    node_colors[solar_idx],      # Solar Radiation -> Hot sources
    node_colors[sky_idx],        # Sky Radiation -> Hot sources
    node_colors[conduction_idx], # Conduction -> Cold sources
    node_colors[convection_idx], # Convection -> Cold sources
    node_colors[floor_idx],      # Floor Radiation -> Cold sources
    node_colors[crop_idx],       # Heat Exchange with Crops -> Cold sources
    node_colors[fog_idx],        # Heat Exchange with Fog -> Cold sources
    node_colors[cover_idx],      # Cover Radiation -> Cold sources
    node_colors[cover_idx],      # Cover Radiation -> Hot sources
    node_colors[cold_idx],       # Cold sources -> Net Heaflow
    node_colors[hot_idx]         # Hot sources -> Net Heaflow
]

# Create the Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(

    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels_sankey,
        color=node_colors,
    ),
    link=dict(
        source=source,  # indices correspond to labels
        target=target,
        value=value,
        color=link_colors
    ))])

fig_sankey.update_layout(
    font=dict(
        shadow="None",
        size=16,           
        color="white",
        #family="Arial" 
    ),
    height=500
)

st.subheader("Heat Flow Sankey Diagram")
st.plotly_chart(fig_sankey, use_container_width=True)

# Display calculated heat flows
st.subheader("Calculated Heat Flows")
st.write(f"**Conduction (Q_cond):** {Q_cond_val:.2f} W")
st.write(f"**Convection (Q_conv):** {Q_conv_val:.2f} W")
st.write(f"**Solar Radiation (Q_solar):** {Q_solar_val:.2f} W")
st.write(f"**Floor Radiation (Q_floor_rad):** {Q_floor_rad_val:.2f} W")
st.write(f"**Sky Radiation (Q_long_sky):** {Q_sky_val:.2f} W")
st.write(f"**Cover Radiation (Q_long_cover):** {Q_cover_val:.2f} W")
st.write(f"**Heat Exchange with Crops (Q_crop):** {Q_crop_val:.2f} W")
st.write(f"**Heat Exchange with Fog (Q_fog):** {Q_fog_val:.2f} W")
st.write(f"**Maximum Latent Heat Exchange (Q_lat):** {Q_lat_val:.2f} W")
st.write(f"**Maximum Sensible Heat Exchange (Q_sens):** {Q_sens_val:.2f} W")


if st.sidebar.button("💾 Export Heat Flows as CSV"):
    df_export = pd.DataFrame({
        'Flow': list(flows.keys()),
        'Value (W)': list(flows.values())
    })
    csv = df_export.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name='heat_flows.csv',
        mime='text/csv'
    )

