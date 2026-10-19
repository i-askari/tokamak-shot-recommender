"""The physics parametrization: which signals become which feature columns.

Every raw diagnostic time series is reduced to a peak (95th percentile) and a
mean over the plasma window, so one shot becomes one fixed-length row.
"""

NBI_BEAMS = ["NB15L", "NB15R", "NB21L", "NB21R", "NB30L", "NB30R", "NB33L", "NB33R"]

ECH_TOWERS = ["BORIS", "CHEWBACCA", "DOROTHY", "HAN", "KATYA", "LEIA",
              "LION", "LUKE", "NASA", "NATASHA", "R2D2", "SCARECROW"]

GAS_VALVES = ["GASA", "GASB", "GASC", "GASD", "GASE", "LOB1", "LOB2",
              "PFX1", "PFX2", "PFX3", "UOB"]

CORE_COLUMNS = [
    "peak_temp", "mean_temp",
    "peak_density", "mean_density",
    "peak_ip", "mean_ip",
    "kappa", "tritop", "tribot", "betan", "q95",
    "peak_ts_tangential_density", "mean_ts_tangential_density",
    "peak_ts_tangential_temp", "mean_ts_tangential_temp",
]

NBI_BEAM_COLUMNS = [f"{stat}_pabs_{beam}" for beam in NBI_BEAMS for stat in ("peak", "mean")]
ECH_COLUMNS = [f"{stat}_echpwr_{tower}" for tower in ECH_TOWERS for stat in ("peak", "mean")]
GASFLOW_COLUMNS = [f"{stat}_gasflow_{valve}" for valve in GAS_VALVES for stat in ("peak", "mean")]

# The 77-dimensional vector used for similarity search.
PARAM_COLUMNS = CORE_COLUMNS + NBI_BEAM_COLUMNS + ECH_COLUMNS + GASFLOW_COLUMNS

# Human-readable catalog, used by the docs and the web demo.
PARAMETER_CATALOG = {
    "Kinetic profiles (Thomson scattering)": [
        ("peak_temp / mean_temp", "Core electron temperature T_e", "eV"),
        ("peak_density / mean_density", "Core electron density n_e", "m^-3"),
        ("*_ts_tangential_temp / *_density", "Tangential-view T_e and n_e", "eV, m^-3"),
    ],
    "Plasma current": [
        ("peak_ip / mean_ip", "Plasma current I_p", "A"),
    ],
    "Equilibrium shape (EFIT)": [
        ("kappa", "Elongation", "-"),
        ("tritop / tribot", "Upper / lower triangularity", "-"),
        ("betan", "Normalized beta (pressure / magnetic pressure)", "-"),
        ("q95", "Safety factor at 95% flux surface", "-"),
    ],
    "Neutral beam heating (8 beams)": [
        ("peak_pabs_<beam> / mean_pabs_<beam>", "Absorbed NBI power per beam", "W"),
    ],
    "Electron cyclotron heating (12 gyrotrons)": [
        ("peak_echpwr_<gyrotron> / mean_echpwr_<gyrotron>", "Forward ECH power per gyrotron", "W"),
    ],
    "Fueling (11 gas valves)": [
        ("peak_gasflow_<valve> / mean_gasflow_<valve>", "Gas flow per valve", "Torr·L/s"),
    ],
}
