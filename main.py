import pandas as pd
import numpy as np
from math import *
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from scipy.optimize import fsolve

from utils import * 

def thermal_bilan(T_i, T_0, I_global, fog_enabled):
    A_cover = st.session_state.A_cover
    A_floor = st.session_state.A_floor
    D_air = st.session_state.D_air

    Q_cond_val = Q_cond(T_i, T_0, A_cover) # Conduction heat transfer U*A*(T_i-T_0)
    Q_conv_val = Q_conv(T_i, T_0, A_cover) # Convection heat transfer h*A*(T_i-T_0)
    Q_cover_val = Q_long_cover(T_i, A_cover) # Long-wavelength radiation heat transfer between the greenhouse and the cover epsilon_i*sigma*A*(T_i^4)
    Q_crop_val = Q_crop(T_i, A_floor) # Heat exchange with crops 2*A_floor*LAI*rho_air*c_air*(T_i-T_plant)/r_b
    Q_fog_val = Q_fog(T_i, st.session_state.T_target, st.session_state.delta_t) if fog_enabled else 0 # Heat exchange with fog Q_sens + Q_lat
    Q_solar_val = Q_solar(I_global, A_floor) # Solar heat transfer tau_cover*A_floor*I_global
    Q_floor_val = Q_long_floor(st.session_state.T_floor) # Floor radiation epsilon_floor*sigma*A_floor*f_sol*(T_floor^4)
    Q_sky_val = Q_long_sky(T_0, A_cover) # Sky radiation epsilon_sky*sigma*A_cover*tau_cover*(T_sky^4)
    Q_vent_val = Q_vent_simple(T_0, T_i, D_air) # Ventilation heat 

    Q_gain = Q_floor_val + Q_solar_val + Q_sky_val + 0.5*Q_cover_val
    Q_losses = Q_cond_val + Q_conv_val + 0.5*Q_cover_val + Q_crop_val + Q_fog_val + Q_vent_val
    delta = Q_gain - Q_losses
    return delta

def estimate_temperature(T_0, daytime, fog_enabled):
    if daytime:
        T_increase = np.random.uniform(5, 15)  # Typical increase during the day
    else:
        T_increase = np.random.uniform(2, 5)  # Typical increase at night
    
    if fog_enabled:
        T_increase *= 0.4  # 40% lower temperature increase with fogging
    
    return T_0 + T_increase

# Initialize session_state with default values
for key, value in DEFAULT_PARAMS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("My Tiny Greenhouse Model")
st.sidebar.header("Greenhouse parameters")
st.sidebar.number_input("Cover Surface Area (A_cover) [m²]", min_value=1.0, key='A_cover', step=10.0)
st.sidebar.number_input("Floor Surface Area (A_floor) [m²]", min_value=1.0, key='A_floor', step=10.0)
st.sidebar.number_input("Misting period (delta_t) [s]", min_value=1.0, key='delta_t', step=10.0)

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

    times = np.linspace(0, 24, int(24 / 0.5) + 1)

    # Linear interpolation of outdoor temperature (T_0)
    T_0 = np.interp(hour_mod, times, temperatures)

    # Calculation of global solar radiation (I_global)
    sunrise = 6.0
    sunset = 18.0
    if sunrise <= hour_mod <= sunset:
        # Calculate solar angle
        solar_angle = np.pi * (hour_mod - sunrise) / (sunset - sunrise)
        I_global = 1000.0 * np.sin(solar_angle) # Maximum intensity of 1000 W/m²
        if I_global < 0:
            I_global = 0.0
    else:
        I_global = 0.0  # No solar radiation at night
    
    st.session_state.T_0 = T_0
    st.session_state.T_i = estimate_temperature(st.session_state.T_i, 6 <= hour_mod <= 18, True)
    st.session_state.T_cover = st.session_state.T_i  # Outdoor wall temperature

    st.session_state.I_global = I_global

    st.session_state.prev_time_str = st.session_state.time_str

T_int = fsolve(thermal_bilan, 30, args=(st.session_state.T_0, st.session_state.I_global, True))[0]
st.write(f"**Estimated inside temperature:** {T_int:.2f} °C")

st.sidebar.header("Temperature and solar radiation")
st.sidebar.slider("Global solar radiation [$W/m^2$]", 0.0, 1360.0, key='I_global')
#st.sidebar.slider("Indoor temperature (T_i) [°C]", -10.0, 50.0, key='T_i')
st.sidebar.slider("Outdoor temperature [$°C$]", -10.0, 50.0, key='T_0')
#st.sidebar.slider("Outdoor wall temperature [°C]", -10.0, 50.0, key='T_cover')
#st.sidebar.slider("Sky temperature [°C]", -10.0, 20.0, key='T_sky')
#st.sidebar.slider("Floor temperature [°C]", -10.0, 50.0, key='T_floor')
st.sidebar.slider("Fog water temperature [$°C$]", 0.0, 40.0, key='T_target')
st.sidebar.slider("Ventilation air flow rate [$kg/s$]", 0.0, 100.0, key='D_air')

T_i = st.session_state['T_i']
T_0 = st.session_state['T_0']
T_cover = st.session_state['T_cover']
T_floor = st.session_state['T_floor']
T_target = st.session_state['T_target']
A_cover = st.session_state['A_cover']
A_floor = st.session_state['A_floor']
I_global = st.session_state['I_global']
delta_t = st.session_state['delta_t']
D_air = st.session_state['D_air']

# Calculate heat transfer rates
Q_cond_val = Q_cond(T_cover, T_0, A_cover)
Q_conv_val = Q_conv(T_cover, T_0, A_cover)
Q_cond_conv_val = abs(Q_cond_val + Q_conv_val)
Q_solar_val = Q_solar(I_global, A_floor)
Q_floor_rad_val = Q_long_floor(T_floor)
Q_sky_val = Q_long_sky(T_0, A_cover)
Q_cover_val = Q_long_cover(T_i, A_cover)
Q_long_val = Q_sky_val + Q_cover_val + Q_floor_rad_val
Q_crop_val = Q_crop(T_i, A_floor)
Q_lat_val = Q_lat(delta_t)
Q_sens_val = Q_sens(T_i, T_target) 
Q_fog_val = Q_fog(T_i, T_target, delta_t)
Q_vent_val = abs(Q_vent_simple(T_0, T_i, D_air))
net_heat = Q_cond_conv_val + Q_solar_val + Q_long_val + Q_crop_val + Q_fog_val + Q_floor_rad_val + Q_vent_val

flows = {
    'Conduction & Convection': Q_cond_conv_val,
    'Solar Radiation': Q_solar_val,
    'IR Radiation (Floor, Sky, Cover)': Q_long_val,
    'Crops evapotranspiration': Q_crop_val,
    'Fog': Q_fog_val,
    'Ventilation': Q_vent_val,
}

# Initialize the 'Others' sum
others_value = 0

flows_pie = {}

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
    'Ventilation',
    'Cold sources',
    'Hot sources',
    'Net Heaflow'
]

solar_idx = labels_sankey.index('Solar Radiation')
sky_idx = labels_sankey.index('Sky Radiation')
conduction_idx = labels_sankey.index('Conduction')
convection_idx = labels_sankey.index('Convection')
cover_idx = labels_sankey.index('Cover Radiation')
floor_idx = labels_sankey.index('Floor Radiation')
crop_idx = labels_sankey.index('Heat Exchange with Crops')
fog_idx = labels_sankey.index('Heat Exchange with Fog')
vent_idx = labels_sankey.index('Ventilation')
cold_idx = labels_sankey.index('Cold sources')
hot_idx = labels_sankey.index('Hot sources')
net_idx = labels_sankey.index('Net Heaflow')

source = [
    solar_idx, sky_idx, conduction_idx, convection_idx,
    floor_idx, crop_idx, fog_idx, vent_idx,
    cover_idx, cover_idx,  
    cold_idx, hot_idx
]

target = [
    hot_idx, hot_idx, cold_idx, cold_idx,
    hot_idx, cold_idx, cold_idx, cold_idx,
    cold_idx, hot_idx,  
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
    Q_vent_val,           # Ventilation -> Cold sources
    Q_cover_val / 2,      # Cover Radiation -> Cold sources
    Q_cover_val / 2,      # Cover Radiation -> Hot sources
    None,                 # Cold sources -> Net Heaflow 
    None                  # Hot sources -> Net Heaflow 
]

cold_flow_total = Q_cond_val + Q_conv_val + Q_crop_val + Q_fog_val + (Q_cover_val / 2) + Q_vent_val
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
    '#e9ff00',   # Ventilation
    '#0061ff',   # Cold sources
    '#e20f0f',   # Hot sources
    '#808080',   # Net Heaflow
]

link_colors = [
    node_colors[solar_idx],      # Solar Radiation -> Hot sources
    node_colors[sky_idx],        # Sky Radiation -> Hot sources
    node_colors[conduction_idx], # Conduction -> Cold sources
    node_colors[convection_idx], # Convection -> Cold sources
    node_colors[floor_idx],      # Floor Radiation -> Cold sources
    node_colors[crop_idx],       # Heat Exchange with Crops -> Cold sources
    node_colors[fog_idx],        # Heat Exchange with Fog -> Cold sources
    node_colors[vent_idx],       # Ventilation -> Cold sources
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
        size=16,           
        color="black",
        #family="Arial" 
    ),
    height=500
)

st.subheader("Heat Flow Sankey Diagram")
st.plotly_chart(fig_sankey, use_container_width=True)

# times_split = np.linspace(0, 24, 100) 
times_slice = np.linspace(0, 24, int(24 / 0.5) + 1)

temperatures_fog = []
temperatures_no_fog = []

for t in times_slice:
    T_0 = temperatures[int(t * 2)]
    sunrise = 6.0
    sunset = 18.0
    if sunrise <= t <= sunset:
        # Calculate solar angle
        solar_angle = np.pi * (t - sunrise) / (sunset - sunrise)
        I_global = 1000.0 * np.sin(solar_angle) # Maximum intensity of 1000 W/m²
        if I_global < 0:
            I_global = 0.0
    else:
        I_global = 0.0
    
    T_i_fog = fsolve(thermal_bilan, 30, args=(T_0, I_global, True))[0]
    T_i_no_fog = fsolve(thermal_bilan, 30, args=(T_0, I_global, False))[0]

    temperatures_fog.append(T_i_fog)
    temperatures_no_fog.append(T_i_no_fog)


estimate_temperatures_no_fog = [
    estimate_temperature(T, 6 <= t <= 18, False) for t, T in zip(times_slice, temperatures)
]
estimate_temperatures_fog = [
    estimate_temperature(T, 6 <= t <= 18, True) for t, T in zip(times_slice, temperatures)
]

st.subheader("Greenhouse Temperature Simulation")

plt.figure(figsize=(10, 6))
plt.plot(times_slice, temperatures_fog, label="model with fogging", color="blue")
plt.plot(times_slice, temperatures_no_fog, label="model without fogging", color="red")
plt.plot(times_slice, estimate_temperatures_fog, label="estimated with fogging", linestyle="--", color="blue")
plt.plot(times_slice, estimate_temperatures_no_fog, label="estimated without fogging", linestyle="--", color="red")
plt.xlabel("Time (h)")
plt.ylabel("Inside temperature (°C)")
plt.legend()
plt.grid()
plt.xlim(0, 24)  
plt.xticks(np.arange(0, 25, 2))  
st.pyplot(plt)

# Display calculated heat flows
st.subheader("Calculated Heat Flows")
st.write(f"**Conduction (Q_cond):** {Q_cond_val:.2f} W")
st.write(f"**Convection (Q_conv):** {Q_conv_val:.2f} W")
st.write(f"**Solar Radiation (Q_solar):** {Q_solar_val:.2f} W")
st.write(f"**Floor Radiation (Q_long_floor):** {Q_floor_rad_val:.2f} W")
st.write(f"**Sky Radiation (Q_long_sky):** {Q_sky_val:.2f} W")
st.write(f"**Cover Radiation (Q_long_cover):** {Q_cover_val:.2f} W")
st.write(f"**Heat Exchange with Crops (Q_crop):** {Q_crop_val:.2f} W")
st.write(f"**Heat Exchange with Fog (Q_fog):** {Q_fog_val:.2f} W")
st.write(f"**Maximum Latent Heat Exchange (Q_lat):** {Q_lat_val:.2f} W")
st.write(f"**Maximum Sensible Heat Exchange (Q_sens):** {Q_sens_val:.2f} W")
st.write(f"**Ventilation Heat Flow (Q_vent):** {Q_vent_val:.2f} W")


heat_flows = {
    'Conduction (Q_cond)': Q_cond_val,
    'Convection (Q_conv)': Q_conv_val,
    'Solar Radiation (Q_solar)': Q_solar_val,
    'Floor Radiation (Q_long_floor)': Q_floor_rad_val,
    'Sky Radiation (Q_long_sky)': Q_sky_val,
    'Cover Radiation (Q_long_cover)': Q_cover_val,
    'Heat Exchange with Crops (Q_crop)': Q_crop_val,
    'Heat Exchange with Fog (Q_fog)': Q_fog_val,
    'Maximum Latent Heat Exchange (Q_lat)': Q_lat_val,
    'Maximum Sensible Heat Exchange (Q_sens)': Q_sens_val
}

df_export = pd.DataFrame({
    'Flow': list(heat_flows.keys()),
    'Value (W)': list(heat_flows.values())
})
csv = df_export.to_csv(index=False)

st.sidebar.download_button(
    label="💾 Export Heat Flows as CSV",
    data=csv,
    file_name='heat_flows.csv',
    mime='text/csv'
)

