import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GeoRes3D – Model 3D Resistivitas",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS – LIGHT MODERN PROFESSIONAL
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%);
        background-attachment: fixed;
    }

    /* ── HERO HEADER ── */
    .hero-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 35%, #1565c0 65%, #0277bd 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(26,35,126,0.25);
        animation: fadeSlideDown 0.7s ease-out;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -40px;
        width: 250px; height: 250px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin: 0 0 1.2rem 0;
        font-weight: 400;
        line-height: 1.6;
        max-width: 700px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 8px;
        backdrop-filter: blur(5px);
    }

    /* ── METRIC CARDS ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin: 1.5rem 0;
        animation: fadeSlideUp 0.6s ease-out 0.2s both;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        border: 1px solid rgba(255,255,255,0.8);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1a237e, #0277bd);
        border-radius: 16px 16px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(26,35,126,0.15);
    }
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
        display: block;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a237e;
        display: block;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #7986cb;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
        display: block;
    }

    /* ── SECTION CARDS ── */
    .section-card {
        background: white;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        border: 1px solid rgba(230,235,255,0.8);
        margin-bottom: 1.5rem;
        animation: fadeSlideUp 0.5s ease-out;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #e8eaf6, transparent);
        margin-left: 8px;
    }

    /* ── INFO BANNER ── */
    .geo-banner {
        background: linear-gradient(135deg, #e8eaf6, #e3f2fd);
        border-left: 4px solid #3f51b5;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #283593;
        animation: fadeIn 0.4s ease-out;
    }
    .note-banner {
        background: linear-gradient(135deg, #fff8e1, #fff3e0);
        border-left: 4px solid #ff9800;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #e65100;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(232,234,246,0.5);
        border-radius: 14px;
        padding: 5px;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #5c6bc0 !important;
        transition: all 0.2s ease !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a237e, #1565c0) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(26,35,126,0.3) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0 0 0 !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 40%, #1565c0 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.92) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stTextArea label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { color: white !important; }
    [data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stTextArea textarea::placeholder {
        color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.1) !important;
        border: 2px dashed rgba(255,255,255,0.35) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background: white !important;
    }
    /* Sidebar section headers */
    .sidebar-section {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 8px 12px;
        margin: 12px 0 8px 0;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255,255,255,0.7) !important;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sidebar-logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: white !important;
        letter-spacing: -0.5px;
    }
    .sidebar-logo-sub {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.6) !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── EXPANDER ── */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    /* ── SPINNER & STATUS ── */
    .stSpinner > div { border-top-color: #3f51b5 !important; }
    .success-banner {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 4px solid #43a047;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        color: #2e7d32;
        font-weight: 500;
        animation: slideInLeft 0.4s ease-out;
    }

    /* ── DATAFRAME ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
    .stDataFrame table { font-size: 0.85rem !important; }

    /* ── SELECT SLIDER ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #1a237e !important;
        border-color: #1a237e !important;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] { color: #5c6bc0; }

    /* ── CAPTION ── */
    .stCaption { color: #7986cb !important; font-style: italic; font-size: 0.8rem !important; }

    /* ── WELCOME SCREEN ── */
    .welcome-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        border-top: 3px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeSlideUp 0.5s ease-out both;
    }
    .feature-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.12); }
    .feature-card:nth-child(1) { border-top-color: #3f51b5; animation-delay: 0.1s; }
    .feature-card:nth-child(2) { border-top-color: #0277bd; animation-delay: 0.2s; }
    .feature-card:nth-child(3) { border-top-color: #00897b; animation-delay: 0.3s; }
    .feature-card:nth-child(4) { border-top-color: #f57c00; animation-delay: 0.4s; }
    .feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 600; color: #1a237e; font-size: 0.95rem; margin-bottom: 0.3rem; }
    .feature-desc { font-size: 0.83rem; color: #78909c; line-height: 1.5; }

    /* ── UPLOAD ZONE LABEL ── */
    .upload-hint {
        background: linear-gradient(135deg, #e8eaf6, #e3f2fd);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px dashed #7986cb;
    }
    .upload-hint-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .upload-hint-text { font-size: 0.9rem; color: #3949ab; font-weight: 500; }

    /* ── PLOT CONTAINER ── */
    .plot-container {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }

    /* ── FOOTER ── */
    .app-footer {
        text-align: center;
        padding: 1.5rem;
        color: #9fa8da;
        font-size: 0.78rem;
        border-top: 1px solid #e8eaf6;
        margin-top: 2rem;
    }

    /* ── ANIMATIONS ── */
    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-15px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.04); }
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TELFORD RESISTIVITY TABLE (Ω·m)
# Sumber: Telford et al., 1990 – Applied Geophysics
# ─────────────────────────────────────────────
TELFORD_TABLE = {
    "Udara / Dry Gas":              (1e6,  1e12, "Udara / Dry Gas",              "Non-konduktif"),
    "Granit":                       (200,  1e5,  "Granit",                       "Batuan beku asam"),
    "Andesit / Basalt":             (10,   1e5,  "Andesit / Basalt",             "Batuan beku mafik"),
    "Riolit":                       (500,  1e5,  "Riolit",                       "Batuan beku vulkanik asam"),
    "Tuf Vulkanik":                 (20,   2000, "Tuf Vulkanik",                 "Endapan piroklastik"),
    "Batu Gamping (Limestone)":     (500,  1e7,  "Batu Gamping",                 "Batuan karbonat"),
    "Batu Gamping Terkarstifikasi": (100,  1e5,  "Batu Gamping Terkarstifikasi", "Karbonat dengan rongga"),
    "Dolomit":                      (100,  1e4,  "Dolomit",                      "Batuan karbonat Mg"),
    "Batupasir (Sandstone)":        (1,    1e4,  "Batupasir",                    "Batuan sedimen klastik kasar"),
    "Batupasir Jenuh Air":          (1,    100,  "Batupasir Jenuh Air",          "Batupasir tersaturasi"),
    "Batulempung (Claystone)":      (1,    100,  "Batulempung",                  "Batuan sedimen klastik halus"),
    "Serpih (Shale)":               (5,    500,  "Serpih (Shale)",               "Batuan sedimen terkonsolidasi"),
    "Lempung (Clay)":               (1,    100,  "Lempung",                      "Tanah berbutir halus plastis"),
    "Lanau (Silt)":                 (10,   200,  "Lanau",                        "Tanah berbutir halus"),
    "Pasir Kering (Dry Sand)":      (100,  1e5,  "Pasir Kering",                 "Sedimen tidak jenuh"),
    "Pasir Basah (Wet Sand)":       (10,   500,  "Pasir Basah",                  "Sedimen tersaturasi"),
    "Kerikil / Gravel":             (100,  600,  "Kerikil / Gravel",             "Sedimen kasar"),
    "Gambut (Peat)":                (5,    100,  "Gambut",                       "Endapan organik lahan basah"),
    "Tanah Humus":                  (10,   150,  "Tanah Humus",                  "Lapisan tanah atas organik"),
    "Air Tawar (Freshwater)":       (10,   100,  "Air Tawar",                    "Akuifer air tawar"),
    "Air Asin (Saltwater)":         (0.01, 1,    "Air Asin",                     "Akuifer air payau/asin"),
    "Lempung Jenuh Garam":          (0.01, 1,    "Lempung Jenuh Garam",          "Sedimen salin"),
    "Alluvium (Campuran)":          (10,   800,  "Alluvium",                     "Endapan sungai campuran"),
    "Batubara (Coal)":              (200,  1e4,  "Batubara",                     "Batuan sedimen organik"),
    "Kuarsit":                      (500,  5e4,  "Kuarsit",                      "Batuan metamorf"),
    "Filit / Slate":                (10,   2000, "Filit / Slate",                "Batuan metamorf derajat rendah"),
    "Sekis (Schist)":               (20,   1e4,  "Sekis",                        "Batuan metamorf"),
    "Gneiss":                       (100,  1e5,  "Gneiss",                       "Batuan metamorf derajat tinggi"),
    "Marmer":                       (100,  2.5e8,"Marmer",                       "Batuan metamorf karbonat"),
    "Lempung Terkonsolidasi":       (20,   200,  "Lempung Terkonsolidasi",       "Lempung padat / overconsolidated"),
}

GEOLOGICAL_ENVIRONMENTS = {
    "Alluvium (Endapan Sungai)": {
        "description": "Endapan sungai muda terdiri dari campuran lempung, lanau, pasir, dan kerikil.",
        "lithologies": [
            "Lempung (Clay)", "Lanau (Silt)", "Pasir Basah (Wet Sand)", "Pasir Kering (Dry Sand)",
            "Kerikil / Gravel", "Alluvium (Campuran)", "Air Tawar (Freshwater)", "Gambut (Peat)", "Tanah Humus",
        ],
    },
    "Dataran Pantai / Pesisir": {
        "description": "Endapan pesisir, delta, atau rawa pasang surut.",
        "lithologies": [
            "Lempung (Clay)", "Lanau (Silt)", "Pasir Basah (Wet Sand)", "Gambut (Peat)",
            "Air Asin (Saltwater)", "Lempung Jenuh Garam", "Alluvium (Campuran)", "Tanah Humus",
        ],
    },
    "Lahan Gambut / Rawa": {
        "description": "Lingkungan lahan basah, dominan endapan organik dan lempung jenuh air.",
        "lithologies": [
            "Gambut (Peat)", "Lempung (Clay)", "Lanau (Silt)", "Air Tawar (Freshwater)",
            "Tanah Humus", "Alluvium (Campuran)",
        ],
    },
    "Formasi Sedimen Tersier": {
        "description": "Batuan sedimen Tersier: batupasir, serpih, batulempung, dan batugamping.",
        "lithologies": [
            "Batupasir (Sandstone)", "Batupasir Jenuh Air", "Serpih (Shale)",
            "Batulempung (Claystone)", "Batu Gamping (Limestone)", "Lempung (Clay)",
            "Lempung Terkonsolidasi", "Batubara (Coal)",
        ],
    },
    "Formasi Volkanik": {
        "description": "Produk vulkanisme: lava, tuf, dan endapan lahar.",
        "lithologies": [
            "Andesit / Basalt", "Riolit", "Tuf Vulkanik", "Granit",
            "Lempung (Clay)", "Tanah Humus", "Batupasir (Sandstone)",
        ],
    },
    "Batuan Karbonat / Karst": {
        "description": "Dominasi batugamping dan dolomit, dapat terkarstifikasi.",
        "lithologies": [
            "Batu Gamping (Limestone)", "Batu Gamping Terkarstifikasi", "Dolomit",
            "Air Tawar (Freshwater)", "Lempung (Clay)", "Batupasir (Sandstone)",
        ],
    },
    "Batuan Beku Intrusif": {
        "description": "Lingkungan dengan dominasi batuan beku: granit, diorit, dll.",
        "lithologies": [
            "Granit", "Andesit / Basalt", "Kuarsit",
            "Lempung (Clay)", "Tanah Humus", "Batupasir (Sandstone)",
        ],
    },
    "Batuan Metamorf": {
        "description": "Lingkungan geologi dengan batuan metamorf: sekis, gneiss, filit.",
        "lithologies": [
            "Sekis (Schist)", "Gneiss", "Filit / Slate", "Kuarsit", "Marmer",
            "Lempung (Clay)", "Tanah Humus",
        ],
    },
    "Endapan Danau (Lakustrin)": {
        "description": "Endapan di lingkungan danau: lempung, lanau, dan gambut.",
        "lithologies": [
            "Lempung (Clay)", "Lanau (Silt)", "Gambut (Peat)", "Air Tawar (Freshwater)",
            "Tanah Humus", "Batulempung (Claystone)", "Alluvium (Campuran)",
        ],
    },
    "Universal (Semua Litologi)": {
        "description": "Gunakan semua litologi dari tabel Telford tanpa filter geologi regional.",
        "lithologies": list(TELFORD_TABLE.keys()),
    },
}

# ─────────────────────────────────────────────
# CUSTOM COLORSCALE – IDENTIK DENGAN RES2DINV
# ─────────────────────────────────────────────
RES2DINV_COLORSCALE = [
    [0.000, "#00008B"],
    [0.083, "#0000FF"],
    [0.167, "#007FFF"],
    [0.250, "#00BFFF"],
    [0.333, "#00FFFF"],
    [0.400, "#00C080"],
    [0.450, "#00A000"],
    [0.500, "#40C040"],
    [0.560, "#80D040"],
    [0.620, "#C8D400"],
    [0.667, "#FFFF00"],
    [0.720, "#D4A000"],
    [0.780, "#C87820"],
    [0.833, "#FF6400"],
    [0.890, "#FF0000"],
    [0.940, "#C00000"],
    [1.000, "#800080"],
]

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,255,1)",
    font=dict(family="Inter, sans-serif", color="#37474f"),
    margin=dict(l=10, r=10, t=50, b=10),
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def classify_resistivity(rho, active_lithologies):
    matches = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE:
            continue
        rmin, rmax, label, desc = TELFORD_TABLE[name]
        if rmin <= rho <= rmax:
            center = (np.log10(rmin) + np.log10(rmax)) / 2
            score = abs(np.log10(rho) - center)
            matches.append((score, label, desc, rmin, rmax))
    if not matches:
        best = None; best_dist = 1e18
        for name in active_lithologies:
            if name not in TELFORD_TABLE: continue
            rmin, rmax, label, desc = TELFORD_TABLE[name]
            dist = min(abs(rho - rmin), abs(rho - rmax))
            if dist < best_dist:
                best_dist = dist; best = (label, desc, rmin, rmax)
        if best:
            return [{"lithology": best[0], "description": best[1],
                     "range": f"{best[2]}–{best[3]} Ω·m", "confidence": "Estimasi terdekat"}]
        return [{"lithology": "Tidak teridentifikasi", "description": "-",
                 "range": "-", "confidence": "-"}]
    matches.sort(key=lambda x: x[0])
    results = []
    for i, (score, label, desc, rmin, rmax) in enumerate(matches[:3]):
        confidence = "Tinggi" if i == 0 else ("Sedang" if i == 1 else "Rendah")
        results.append({"lithology": label, "description": desc,
                         "range": f"{rmin}–{rmax} Ω·m", "confidence": confidence})
    return results


def get_primary_lithology(rho, active_lithologies):
    results = classify_resistivity(rho, active_lithologies)
    return results[0]["lithology"] if results else "Tidak teridentifikasi"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    xl = pd.ExcelFile(uploaded_file)
    df_geo = xl.parse("GEOMETRY")
    df_data = xl.parse("DATA")
    df_geo.columns = [c.strip().upper() for c in df_geo.columns]
    df_data.columns = [c.strip().upper() for c in df_data.columns]
    col_map_data = {}
    for c in df_data.columns:
        cu = c.upper()
        if "LINE" in cu:     col_map_data[c] = "LINE"
        elif "DIST" in cu:   col_map_data[c] = "DISTANCE"
        elif "POS" in cu:    col_map_data[c] = "POSITION"
        elif "DEPTH" in cu:  col_map_data[c] = "DEPTH"
        elif "RESIST" in cu: col_map_data[c] = "RESISTIVITY"
    df_data = df_data.rename(columns=col_map_data)
    col_map_geo = {}
    for c in df_geo.columns:
        cu = c.upper()
        if "LINE" in cu: col_map_geo[c] = "LINE"
        elif "Y" in cu or "POS" in cu: col_map_geo[c] = "Y_POSITION"
    df_geo = df_geo.rename(columns=col_map_geo)
    line_y = dict(zip(df_geo["LINE"].astype(str), df_geo["Y_POSITION"]))
    df_data["LINE"] = df_data["LINE"].astype(str)
    df_data["Y"] = df_data["LINE"].map(line_y)
    df_data["X"] = pd.to_numeric(df_data.get("DISTANCE", df_data.get("POSITION", 0)), errors="coerce")
    df_data["Z"] = pd.to_numeric(df_data["DEPTH"], errors="coerce")
    df_data["RHO"] = pd.to_numeric(df_data["RESISTIVITY"], errors="coerce")
    df_data = df_data.dropna(subset=["X", "Y", "Z", "RHO"])
    df_data = df_data[df_data["RHO"] > 0]
    return df_data, line_y


# ─────────────────────────────────────────────
# BUILD 3D GRID
# ─────────────────────────────────────────────
def build_grid(df, nx=60, ny=60, nz=30):
    xi = np.linspace(df["X"].min(), df["X"].max(), nx)
    yi = np.linspace(df["Y"].min(), df["Y"].max(), ny)
    zi = np.linspace(df["Z"].min(), df["Z"].max(), nz)
    XX, YY, ZZ = np.meshgrid(xi, yi, zi, indexing="ij")
    pts = df[["X", "Y", "Z"]].values
    vals = df["RHO"].values
    grid = griddata(pts, vals, (XX, YY, ZZ), method="linear")
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nn = griddata(pts, vals, (XX, YY, ZZ), method="nearest")
        grid[nan_mask] = grid_nn[nan_mask]
    return xi, yi, zi, grid


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_3d_volume(xi, yi, zi, grid, vmin, vmax, title_suffix=""):
    fig = go.Figure(data=go.Volume(
        x=np.repeat(xi, len(yi) * len(zi)),
        y=np.tile(np.repeat(yi, len(zi)), len(xi)),
        z=np.tile(zi, len(xi) * len(yi)),
        value=grid.flatten(),
        isomin=vmin, isomax=vmax,
        opacity=0.15, surface_count=20,
        colorscale=RES2DINV_COLORSCALE,
        colorbar=dict(title="Resistivitas<br>(Ω·m)", tickformat=".1f",
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1,
                      len=0.75, thickness=18),
        cmin=vmin, cmax=vmax,
    ))
    fig.update_layout(
        title=dict(text=f"<b>Model 3D Resistivitas</b>{title_suffix}",
                   font=dict(size=15, color="#1a237e", family="Space Grotesk")),
        scene=dict(
            xaxis=dict(title="Jarak (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(232,234,246,0.3)"),
            yaxis=dict(title="Lintasan (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(227,242,253,0.3)"),
            zaxis=dict(title="Kedalaman (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(243,229,245,0.3)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        height=650,
        **PLOTLY_LAYOUT_BASE,
    )
    return fig


def plot_fence_diagram(df, line_y, vmin, vmax):
    fig = go.Figure()
    for line_name, y_pos in sorted(line_y.items(), key=lambda x: x[1]):
        sub = df[df["LINE"] == str(line_name)]
        if sub.empty: continue
        xi_l = np.linspace(sub["X"].min(), sub["X"].max(), 80)
        zi_l = np.linspace(sub["Z"].min(), sub["Z"].max(), 40)
        XI, ZI = np.meshgrid(xi_l, zi_l)
        pts2d = sub[["X", "Z"]].values
        vals2d = sub["RHO"].values
        grid2d = griddata(pts2d, vals2d, (XI, ZI), method="linear")
        nan_m = np.isnan(grid2d)
        if nan_m.any():
            g2 = griddata(pts2d, vals2d, (XI, ZI), method="nearest")
            grid2d[nan_m] = g2[nan_m]
        YI = np.full_like(XI, float(y_pos))
        fig.add_trace(go.Surface(
            x=XI, y=YI, z=ZI,
            surfacecolor=grid2d,
            colorscale=RES2DINV_COLORSCALE,
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title="Resistivitas<br>(Ω·m)", tickformat=".1f",
                          bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1,
                          len=0.75, thickness=18, x=1.02),
            name=f"Lintasan {line_name}",
            opacity=0.93,
        ))
    fig.update_layout(
        title=dict(text="<b>Fence Diagram</b> – Penampang Vertikal Antar Lintasan",
                   font=dict(size=15, color="#1a237e", family="Space Grotesk")),
        scene=dict(
            xaxis=dict(title="Jarak (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(232,234,246,0.3)"),
            yaxis=dict(title="Lintasan (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(227,242,253,0.3)"),
            zaxis=dict(title="Kedalaman (m)", gridcolor="#e8eaf6", showbackground=True,
                       backgroundcolor="rgba(243,229,245,0.3)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        height=650,
        **PLOTLY_LAYOUT_BASE,
    )
    return fig


def plot_horizontal_slice(xi, yi, zi, grid, depth_idx, depth_val, vmin, vmax):
    slice_data = grid[:, :, depth_idx]
    fig = go.Figure(data=go.Heatmap(
        x=xi, y=yi, z=slice_data.T,
        colorscale=RES2DINV_COLORSCALE,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="Resistivitas<br>(Ω·m)", tickformat=".1f",
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1),
    ))
    fig.update_layout(
        title=dict(text=f"<b>Irisan Horizontal</b> – Kedalaman {depth_val:.2f} m",
                   font=dict(size=15, color="#1a237e", family="Space Grotesk")),
        xaxis=dict(title="Jarak (m)", gridcolor="#e8eaf6"),
        yaxis=dict(title="Lintasan (m)", gridcolor="#e8eaf6"),
        height=500,
        **PLOTLY_LAYOUT_BASE,
    )
    return fig


def plot_vertical_slice(xi, zi, grid, line_idx, line_val, vmin, vmax):
    slice_data = grid[:, line_idx, :]
    fig = go.Figure(data=go.Heatmap(
        x=xi, y=zi, z=slice_data.T,
        colorscale=RES2DINV_COLORSCALE,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="Resistivitas<br>(Ω·m)", tickformat=".1f",
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1),
    ))
    fig.update_layout(
        title=dict(text=f"<b>Irisan Vertikal</b> – Lintasan Y = {line_val:.1f} m",
                   font=dict(size=15, color="#1a237e", family="Space Grotesk")),
        xaxis=dict(title="Jarak (m)", gridcolor="#e8eaf6"),
        yaxis=dict(title="Kedalaman (m)", gridcolor="#e8eaf6", autorange="reversed"),
        height=500,
        **PLOTLY_LAYOUT_BASE,
    )
    return fig


def build_lithology_legend(active_lithologies, vmin, vmax):
    rows = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE: continue
        rmin, rmax, label, desc = TELFORD_TABLE[name]
        if rmax < vmin or rmin > vmax: overlap = "–"
        else:
            lo = max(rmin, vmin); hi = min(rmax, vmax)
            overlap = f"{lo:.1f} – {hi:.1f}"
        rows.append({"Litologi": label, "Rentang Telford (Ω·m)": f"{rmin} – {rmax}",
                     "Terdapat di Data": overlap, "Keterangan": desc})
    return pd.DataFrame(rows)


def build_stats_table(df, active_lithologies):
    bins = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE: continue
        rmin, rmax, label, _ = TELFORD_TABLE[name]
        mask = (df["RHO"] >= rmin) & (df["RHO"] <= rmax)
        cnt = mask.sum()
        if cnt > 0:
            bins.append({"Litologi": label, "Rentang (Ω·m)": f"{rmin} – {rmax}",
                         "Jumlah Titik": cnt, "% Data": f"{100*cnt/len(df):.1f}%",
                         "Rata-rata (Ω·m)": f"{df.loc[mask,'RHO'].mean():.2f}",
                         "Min (Ω·m)": f"{df.loc[mask,'RHO'].min():.2f}",
                         "Max (Ω·m)": f"{df.loc[mask,'RHO'].max():.2f}"})
    return pd.DataFrame(bins).sort_values("Jumlah Titik", ascending=False) if bins else pd.DataFrame()


# ─────────────────────────────────────────────
# METRIC CARD HTML
# ─────────────────────────────────────────────
def metric_card(icon, value, label):
    return f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <span class="metric-value">{value}</span>
        <span class="metric-label">{label}</span>
    </div>"""


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    inject_css()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-text">🌍 GeoRes3D</div>
            <div class="sidebar-logo-sub">Geoelectric 3D Modelling</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="sidebar-section">📂 Upload Data</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("File .xlsx (sheet: GEOMETRY & DATA)", type=["xlsx"],
                                    label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<div class="sidebar-section">🗺️ Geologi Regional</div>', unsafe_allow_html=True)
        env_choice = st.selectbox(
            "Pilih Lingkungan Geologi",
            list(GEOLOGICAL_ENVIRONMENTS.keys()), index=0,
            help="Menyaring litologi yang relevan dari tabel Telford sesuai kondisi geologi lapangan.",
        )
        env_info = GEOLOGICAL_ENVIRONMENTS[env_choice]
        st.markdown(f'<div class="geo-banner">📋 {env_info["description"]}</div>', unsafe_allow_html=True)

        custom_notes = st.text_area(
            "Catatan Geologi (opsional)",
            placeholder="Contoh: Singkapan batugamping di utara, muka air tanah ~2 m...",
            height=75,
        )

        with st.expander("🪨 Litologi Aktif"):
            for lith in env_info["lithologies"]:
                if lith in TELFORD_TABLE:
                    rmin, rmax, label, desc = TELFORD_TABLE[lith]
                    st.markdown(f"**{label}** · {rmin}–{rmax} Ω·m")

        st.markdown("---")
        st.markdown('<div class="sidebar-section">🔧 Resolusi Grid</div>', unsafe_allow_html=True)
        nx = st.slider("Grid X", 20, 100, 60, 10)
        ny = st.slider("Grid Y", 10, 60, 30, 5)
        nz = st.slider("Grid Z", 10, 50, 25, 5)

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.72rem;color:rgba(255,255,255,0.45);text-align:center;line-height:1.5">'
            '📖 Telford et al. (1990)<br><i>Applied Geophysics</i><br>Cambridge Univ. Press</p>',
            unsafe_allow_html=True
        )

    # ── HERO HEADER ──
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">🌍 Model 3D Resistivitas Geolistrik</div>
        <div class="hero-subtitle">
            Visualisasi data geolistrik 2D multi-lintasan menjadi model 3D resistivitas
            interaktif dengan interpretasi litologi berbasis <b>Tabel Telford et al. (1990)</b>
            dan kondisi geologi regional lokasi kajian.
        </div>
        <span class="hero-badge">⚡ Res2DInv Colorscale</span>
        <span class="hero-badge">🪨 30 Jenis Litologi</span>
        <span class="hero-badge">🧊 3D Volume + Fence Diagram</span>
        <span class="hero-badge">📐 Irisan H & V</span>
    </div>
    """, unsafe_allow_html=True)

    # ── WELCOME SCREEN ──
    if uploaded is None:
        st.markdown("""
        <div class="upload-hint">
            <div class="upload-hint-icon">📂</div>
            <div class="upload-hint-text">Upload file Excel (.xlsx) di sidebar kiri untuk memulai</div>
            <div style="font-size:0.78rem;color:#7986cb;margin-top:6px">Sheet yang dibutuhkan: <b>GEOMETRY</b> dan <b>DATA</b></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="welcome-grid">
                <div class="feature-card">
                    <div class="feature-icon">🧊</div>
                    <div class="feature-title">Model 3D Volume</div>
                    <div class="feature-desc">Visualisasi volume resistivitas 3D interaktif dengan colorscale Res2DInv</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🏗️</div>
                    <div class="feature-title">Fence Diagram</div>
                    <div class="feature-desc">Penampang vertikal multi-lintasan dalam ruang 3D</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📐</div>
                    <div class="feature-title">Irisan Horizontal & Vertikal</div>
                    <div class="feature-desc">Peta resistivitas pada kedalaman dan lintasan tertentu</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🪨</div>
                    <div class="feature-title">Interpretasi Litologi</div>
                    <div class="feature-desc">Klasifikasi otomatis berdasarkan Tabel Telford & geologi regional</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**📋 Format Sheet GEOMETRY**")
            st.dataframe(pd.DataFrame({"Line": ["L1","L2","L3"], "Y_Position": [0, 10, 20]}),
                         use_container_width=True, hide_index=True)
            st.markdown("**📋 Format Sheet DATA**")
            st.dataframe(pd.DataFrame({
                "Line": ["L1","L1","L2"], "Distance": [1.5, 2.5, 1.5],
                "Depth": [-0.25, -0.5, -0.25], "Resistivity": [1.29, 2.44, 15.3],
            }), use_container_width=True, hide_index=True)

        with st.expander("📊 Tabel Resistivitas Telford – Referensi Lengkap"):
            rows = [{"Litologi": label, "Min (Ω·m)": rmin, "Max (Ω·m)": rmax, "Keterangan": desc}
                    for _, (rmin, rmax, label, desc) in TELFORD_TABLE.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    # ── LOAD DATA ──
    with st.spinner("⏳ Membaca dan memproses data..."):
        try:
            df, line_y = load_data(uploaded)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            return
    if df.empty:
        st.error("❌ Data kosong atau format kolom tidak dikenali.")
        return

    active_lithologies = env_info["lithologies"]
    vmin = float(df["RHO"].quantile(0.02))
    vmax = float(df["RHO"].quantile(0.98))

    # ── METRIC CARDS ──
    st.markdown(f"""
    <div class="success-banner">
        ✅ Data berhasil dimuat dari <b>{uploaded.name}</b> &nbsp;·&nbsp;
        Lingkungan Geologi: <b>{env_choice}</b>
    </div>
    <div class="metric-row">
        {metric_card("📍", f"{len(df):,}", "Total Titik")}
        {metric_card("📏", str(df['LINE'].nunique()), "Jumlah Lintasan")}
        {metric_card("🔵", f"{df['RHO'].min():.2f} Ω·m", "Min Resistivitas")}
        {metric_card("🔴", f"{df['RHO'].max():.2f} Ω·m", "Max Resistivitas")}
        {metric_card("📊", f"{df['RHO'].mean():.2f} Ω·m", "Rata-rata")}
    </div>
    """, unsafe_allow_html=True)

    if custom_notes:
        st.markdown(f'<div class="note-banner">📝 <b>Catatan Geologi:</b> {custom_notes}</div>',
                    unsafe_allow_html=True)

    # ── BUILD GRID ──
    with st.spinner("🔧 Membangun grid 3D..."):
        xi, yi, zi, grid = build_grid(df, nx=nx, ny=ny, nz=nz)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧊  Model 3D Volume",
        "🏗️  Fence Diagram",
        "📐  Irisan Horizontal",
        "📏  Irisan Vertikal",
        "📊  Interpretasi Litologi",
    ])

    with tab1:
        st.markdown('<div class="section-title">🧊 Model 3D Resistivitas</div>', unsafe_allow_html=True)
        fig3d = plot_3d_volume(xi, yi, zi, grid, vmin, vmax, title_suffix=f" – {env_choice}")
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Model volume 3D hasil interpolasi griddata (scipy). Nilai ditampilkan dalam Ω·m (apparent resistivity). Colorscale identik dengan Res2DInv.")

    with tab2:
        st.markdown('<div class="section-title">🏗️ Fence Diagram</div>', unsafe_allow_html=True)
        fig_fence = plot_fence_diagram(df, line_y, vmin, vmax)
        st.plotly_chart(fig_fence, use_container_width=True)
        st.caption("Penampang vertikal setiap lintasan ditampilkan dalam ruang 3D. Nilai resistivitas dalam Ω·m.")

    with tab3:
        st.markdown('<div class="section-title">📐 Irisan Horizontal</div>', unsafe_allow_html=True)
        depth_labels = [f"{z:.2f} m" for z in zi]
        depth_label_sel = st.select_slider("Pilih Kedalaman", options=depth_labels,
                                           value=depth_labels[len(depth_labels) // 2])
        depth_idx = depth_labels.index(depth_label_sel)
        fig_hz = plot_horizontal_slice(xi, yi, zi, grid, depth_idx, zi[depth_idx], vmin, vmax)
        st.plotly_chart(fig_hz, use_container_width=True)
        slice_vals = grid[:, :, depth_idx].flatten()
        slice_vals = slice_vals[~np.isnan(slice_vals)]
        if len(slice_vals) > 0:
            median_val = float(np.median(slice_vals))
            primary = get_primary_lithology(median_val, active_lithologies)
            st.markdown(
                f'<div class="geo-banner">🔍 Kedalaman <b>{zi[depth_idx]:.2f} m</b> &nbsp;·&nbsp; '
                f'Median: <b>{median_val:.2f} Ω·m</b> &nbsp;·&nbsp; '
                f'Interpretasi dominan: <b>{primary}</b></div>',
                unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-title">📏 Irisan Vertikal per Lintasan</div>', unsafe_allow_html=True)
        line_labels = [f"Y = {y:.1f} m" for y in yi]
        line_label_sel = st.select_slider("Pilih Lintasan (Y)", options=line_labels,
                                          value=line_labels[len(line_labels) // 2])
        line_idx = line_labels.index(line_label_sel)
        fig_vt = plot_vertical_slice(xi, zi, grid, line_idx, yi[line_idx], vmin, vmax)
        st.plotly_chart(fig_vt, use_container_width=True)

    with tab5:
        st.markdown(f'<div class="section-title">📊 Interpretasi Litologi – {env_choice}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"Interpretasi berdasarkan **Tabel Telford et al. (1990)** "
            f"dengan filter lingkungan geologi: **{env_choice}**.")

        st.markdown("#### 📋 Tabel Referensi Litologi Aktif")
        legend_df = build_lithology_legend(active_lithologies, vmin, vmax)
        if not legend_df.empty:
            st.dataframe(legend_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Distribusi Litologi dalam Data")
        stats_df = build_stats_table(df, active_lithologies)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            fig_bar = go.Figure(go.Bar(
                x=stats_df["Litologi"],
                y=[float(v.replace("%","")) for v in stats_df["% Data"]],
                marker=dict(
                    color=[float(v.replace("%","")) for v in stats_df["% Data"]],
                    colorscale=[[0,"#c5cae9"],[0.5,"#3f51b5"],[1,"#1a237e"]],
                    showscale=False,
                    line=dict(color="white", width=1.5),
                ),
                text=stats_df["% Data"],
                textposition="outside",
                textfont=dict(size=11, color="#37474f"),
            ))
            fig_bar.update_layout(
                title=dict(text="<b>Distribusi Litologi</b> (% titik data)",
                           font=dict(size=14, color="#1a237e", family="Space Grotesk")),
                xaxis=dict(title="Litologi", tickangle=-30, gridcolor="#f0f0f0"),
                yaxis=dict(title="% Data", gridcolor="#f0f0f0"),
                height=420,
                bargap=0.3,
                **PLOTLY_LAYOUT_BASE,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada litologi yang cocok dengan rentang data dalam lingkungan geologi ini.")

        st.markdown("---")
        st.markdown("#### 🔎 Interpretasi Titik Data (Sampel)")
        n_sample = min(200, len(df))
        sample_df = df.sample(n_sample, random_state=42).copy()
        sample_df["Litologi Utama"] = sample_df["RHO"].apply(
            lambda r: get_primary_lithology(r, active_lithologies))
        display_cols = ["LINE", "X", "Y", "Z", "RHO", "Litologi Utama"]
        available_cols = [c for c in display_cols if c in sample_df.columns]
        rename_map = {"LINE": "Lintasan", "X": "Jarak (m)", "Y": "Y (m)",
                      "Z": "Kedalaman (m)", "RHO": "Resistivitas (Ω·m)"}
        st.dataframe(sample_df[available_cols].rename(columns=rename_map).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan sampel {n_sample} titik dari total {len(df):,} titik.")

        st.markdown("---")
        st.markdown(
            "📖 **Referensi:** Telford, W.M., Geldart, L.P., & Sheriff, R.E. (1990). "
            "*Applied Geophysics* (2nd ed.). Cambridge University Press.")

    # ── FOOTER ──
    st.markdown("""
    <div class="app-footer">
        🌍 <b>GeoRes3D</b> – Geoelectric 3D Resistivity Modelling &nbsp;·&nbsp;
        Powered by <b>Streamlit</b> + <b>Plotly</b> + <b>SciPy</b> &nbsp;·&nbsp;
        Referensi: Telford et al. (1990) <i>Applied Geophysics</i>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
