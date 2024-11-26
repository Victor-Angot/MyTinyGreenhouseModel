import numpy as np
from math import *

# Greenhouse dimensions
L = 50.0
l = 9.0
H = 3.0

DEFAULT_PARAMS = {
    'U': 5,  # global conduction heat transfer coefficient W/m2K (cover interface)
    'sigma': 5.67e-8, # Boltzmann constant W/m2K4
    'A_cover': 2*(L*H + l*H) + L*l,
    'h': 3, # global convection heat transfer coefficient W/m2K (cover interface)
    'tau_cover': 0.83, # transmittance of the cover
    'A_roof': 0.6*L*l,
    'A_side': 0.25*2*(L*H),
    'V_zone': L*l*H, 
    'A_floor': L*l,
    'Wsat': 0.033, # saturated moisture kg/m3
    'Wi': 0.01724, # average moisture kg/m3
    'I_global': 780.0, # global solar radiation W/m2
    'epsilon_i': 0.92, # long-wavelength radiation emissivity of the cover
    'epsilon_floor': 0.95, # long-wavelength radiation emissivity of the floor
    'epsilon_sky': 0.8, # long-wavelength radiation emissivity of the sky
    'LAI': 3.0, # leaf area index
    'rho_air': 1.2, 
    'c_air': 1008.0,
    'D_air': 30.0, # air flow 
    'N': 2, # cloud cover (octas)
    'r_b': 275.0, 
    'cp': 4185, 
    'm_target_star': 0.7, 
    'delta_H' : 2.45e6, # latent heat of vaporization J/kg
    'delta_t' : 450.0,
    'f_sens' : 0.1, # fraction maximal sensible heat 
    'f_lat' : 0.9, # fraction chaleur latente maximal. Représente l'efficacité de l'évaporation des gouttelettes. 
    # Toutes les gouttelettes pulvérisées ne s'évaporent pas instantanément. 
    # Certaines peuvent retomber au sol ou sur les plantes.
    'T_target' : 12.0,
    'T_i': 28.0,          
    'T_0': 31.0,          
    'T_cover': 40.0,      
    'T_sky': 10.0,
    'T_floor': 35.0,
    'hour': "12:00",      
}

temperatures = np.array([
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
    ])

def calculate_sky_temperature(T_0, N):
    """
    Calculate the sky temperature based on the air temperature and the cloud cover
    T_0: Outdoor air temperature (°C)
    N: Cloud cover (octas)
    """
    T_sky = (T_0 - 6) * (N / 8) + 0.055 * T_0**1.5 * (1 - N / 8)
    return T_sky

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

def Q_long_sky(T_0, A_cover, N):
    '''Q_long_sky represents the long-wavelength radiation heat transfer rate between the greenhouse and the sky'''
    T_sky = calculate_sky_temperature(T_0, N)
    return DEFAULT_PARAMS['sigma']*A_cover*DEFAULT_PARAMS['tau_cover']*(DEFAULT_PARAMS['epsilon_sky']*(T_sky+273.15)**4)

def Q_long_cover(T_i, A_cover):
    '''Q_long_cover represents the long-wavelength radiation heat transfer rate between the greenhouse and the cover'''
    return DEFAULT_PARAMS['sigma']*A_cover*DEFAULT_PARAMS['tau_cover']*(DEFAULT_PARAMS['epsilon_i']*(T_i+273.15)**4)

def Q_long_floor(T_floor):
    k = 0.5
    f_sol = exp(-k*DEFAULT_PARAMS['LAI']) # fraction of soil not covered by plants
    return DEFAULT_PARAMS['epsilon_floor']*DEFAULT_PARAMS['sigma']*DEFAULT_PARAMS['A_floor']*f_sol*(T_floor+273.15)**4

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

def Q_vent_simple(T_0, T_i, D_air):
    '''Calculate the simplified ventilation heat flow in a greenhouse.'''
    c_air = DEFAULT_PARAMS['c_air']
    Q_vent = D_air * c_air * (T_i - T_0) 
    return Q_vent
