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
    page_title="Model 3D Resistivitas Geolistrik",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# TELFORD RESISTIVITY TABLE (Ω·m)
# Sumber: Telford et al., 1990 – Applied Geophysics
# ─────────────────────────────────────────────
TELFORD_TABLE = {
    # (min, max, nama_litologi, deskripsi)
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

# ─────────────────────────────────────────────
# GEOLOGICAL ENVIRONMENT DEFINITIONS
# Setiap lingkungan geologi mendefinisikan subset litologi yang mungkin ada
# ─────────────────────────────────────────────
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
# HELPER: classify resistivity against active lithology set
# ─────────────────────────────────────────────
def classify_resistivity(rho, active_lithologies):
    """
    Mencocokkan nilai resistivitas terhadap rentang Telford
    untuk litologi yang aktif (sesuai lingkungan geologi regional).
    Mengembalikan daftar litologi yang cocok, diurutkan berdasarkan
    seberapa 'tengah' nilai rho berada dalam rentang tersebut.
    """
    matches = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE:
            continue
        rmin, rmax, label, desc = TELFORD_TABLE[name]
        if rmin <= rho <= rmax:
            # skor: seberapa dekat ke pusat rentang (log-scale)
            center = (np.log10(rmin) + np.log10(rmax)) / 2
            score = abs(np.log10(rho) - center)
            matches.append((score, label, desc, rmin, rmax))
    if not matches:
        # fallback: cari yang paling dekat
        best = None
        best_dist = 1e18
        for name in active_lithologies:
            if name not in TELFORD_TABLE:
                continue
            rmin, rmax, label, desc = TELFORD_TABLE[name]
            dist = min(abs(rho - rmin), abs(rho - rmax))
            if dist < best_dist:
                best_dist = dist
                best = (label, desc, rmin, rmax)
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


def build_color_scale_for_env(active_lithologies, vmin, vmax):
    """
    Buat colorscale Plotly berdasarkan rentang data aktual.
    """
    # Gunakan colorscale standar yang informatif untuk resistivitas
    return "RdYlGn_r"  # merah=resistivitas rendah (konduktif), hijau=tinggi (resistif)


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
    # normalise column names
    col_map_data = {}
    for c in df_data.columns:
        cu = c.upper()
        if "LINE" in cu:          col_map_data[c] = "LINE"
        elif "DIST" in cu:        col_map_data[c] = "DISTANCE"
        elif "POS" in cu:         col_map_data[c] = "POSITION"
        elif "DEPTH" in cu:       col_map_data[c] = "DEPTH"
        elif "RESIST" in cu:      col_map_data[c] = "RESISTIVITY"
    df_data = df_data.rename(columns=col_map_data)
    col_map_geo = {}
    for c in df_geo.columns:
        cu = c.upper()
        if "LINE" in cu:     col_map_geo[c] = "LINE"
        elif "Y" in cu or "POS" in cu: col_map_geo[c] = "Y_POSITION"
    df_geo = df_geo.rename(columns=col_map_geo)
    # merge Y positions
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
    # fill NaN with nearest
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
        isomin=vmin,
        isomax=vmax,
        opacity=0.15,
        surface_count=20,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Resistivitas (Ω·m)", tickformat=".0f"),
        cmin=vmin,
        cmax=vmax,
    ))
    fig.update_layout(
        title=f"Model 3D Resistivitas{title_suffix}",
        scene=dict(
            xaxis_title="Jarak (m)",
            yaxis_title="Lintasan (m)",
            zaxis_title="Kedalaman (m)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_fence_diagram(df, line_y, vmin, vmax):
    fig = go.Figure()
    colorscale = "RdYlGn_r"
    for line_name, y_pos in sorted(line_y.items(), key=lambda x: x[1]):
        sub = df[df["LINE"] == str(line_name)]
        if sub.empty:
            continue
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
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title="Resistivitas (Ω·m)", tickformat=".0f", x=1.02),
            name=f"Lintasan {line_name}",
            opacity=0.9,
        ))
    fig.update_layout(
        title="Fence Diagram – Penampang Vertikal Antar Lintasan",
        scene=dict(
            xaxis_title="Jarak (m)",
            yaxis_title="Lintasan (m)",
            zaxis_title="Kedalaman (m)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_horizontal_slice(xi, yi, zi, grid, depth_idx, depth_val, vmin, vmax):
    slice_data = grid[:, :, depth_idx]
    fig = go.Figure(data=go.Heatmap(
        x=xi, y=yi,
        z=slice_data.T,
        colorscale="RdYlGn_r",
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="Resistivitas (Ω·m)", tickformat=".0f"),
    ))
    fig.update_layout(
        title=f"Irisan Horizontal – Kedalaman {depth_val:.2f} m",
        xaxis_title="Jarak (m)",
        yaxis_title="Lintasan (m)",
        height=500,
    )
    return fig


def plot_vertical_slice(xi, zi, grid, line_idx, line_val, vmin, vmax):
    slice_data = grid[:, line_idx, :]
    fig = go.Figure(data=go.Heatmap(
        x=xi, y=zi,
        z=slice_data.T,
        colorscale="RdYlGn_r",
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="Resistivitas (Ω·m)", tickformat=".0f"),
    ))
    fig.update_layout(
        title=f"Irisan Vertikal – Lintasan Y = {line_val:.1f} m",
        xaxis_title="Jarak (m)",
        yaxis_title="Kedalaman (m)",
        height=500,
    )
    return fig


# ─────────────────────────────────────────────
# LITHOLOGY LEGEND TABLE
# ─────────────────────────────────────────────
def build_lithology_legend(active_lithologies, vmin, vmax):
    rows = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE:
            continue
        rmin, rmax, label, desc = TELFORD_TABLE[name]
        # check overlap with data range
        if rmax < vmin or rmin > vmax:
            overlap = "–"
        else:
            lo = max(rmin, vmin)
            hi = min(rmax, vmax)
            overlap = f"{lo:.1f} – {hi:.1f}"
        rows.append({
            "Litologi": label,
            "Rentang Telford (Ω·m)": f"{rmin} – {rmax}",
            "Terdapat di Data": overlap,
            "Keterangan": desc,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# STATISTICS & INTERPRETATION TABLE
# ─────────────────────────────────────────────
def build_stats_table(df, active_lithologies):
    bins = []
    for name in active_lithologies:
        if name not in TELFORD_TABLE:
            continue
        rmin, rmax, label, _ = TELFORD_TABLE[name]
        mask = (df["RHO"] >= rmin) & (df["RHO"] <= rmax)
        cnt = mask.sum()
        if cnt > 0:
            bins.append({
                "Litologi": label,
                "Rentang (Ω·m)": f"{rmin} – {rmax}",
                "Jumlah Titik": cnt,
                "% Data": f"{100*cnt/len(df):.1f}%",
                "Rata-rata (Ω·m)": f"{df.loc[mask,'RHO'].mean():.2f}",
                "Min (Ω·m)": f"{df.loc[mask,'RHO'].min():.2f}",
                "Max (Ω·m)": f"{df.loc[mask,'RHO'].max():.2f}",
            })
    return pd.DataFrame(bins).sort_values("Jumlah Titik", ascending=False) if bins else pd.DataFrame()


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.title("🌍 Model 3D Resistivitas Geolistrik")
    st.markdown(
        "Visualisasi data geolistrik 2D multi-lintasan menjadi model 3D resistivitas "
        "dengan interpretasi litologi berdasarkan **Tabel Telford et al. (1990)** "
        "dan kondisi **geologi regional** lokasi kajian."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        # 1) Upload
        st.subheader("📂 Upload Data Excel")
        uploaded = st.file_uploader("File .xlsx (sheet: GEOMETRY & DATA)", type=["xlsx"])

        st.markdown("---")

        # 2) Geologi Regional
        st.subheader("🗺️ Kondisi Geologi Regional")
        env_choice = st.selectbox(
            "Pilih Lingkungan Geologi",
            list(GEOLOGICAL_ENVIRONMENTS.keys()),
            index=0,
            help="Pilih lingkungan geologi yang sesuai dengan lokasi kajian Anda. "
                 "Ini akan menyaring jenis litologi yang relevan dari tabel Telford.",
        )
        env_info = GEOLOGICAL_ENVIRONMENTS[env_choice]
        st.info(f"📋 {env_info['description']}")

        # Custom notes
        custom_notes = st.text_area(
            "Catatan Geologi Tambahan (opsional)",
            placeholder="Contoh: Terdapat singkapan batugamping di utara lokasi, kedalaman muka air tanah ~2 m...",
            height=80,
        )

        # Show active lithologies
        with st.expander("🪨 Litologi Aktif untuk Lingkungan Ini"):
            for lith in env_info["lithologies"]:
                if lith in TELFORD_TABLE:
                    rmin, rmax, label, desc = TELFORD_TABLE[lith]
                    st.markdown(f"- **{label}**: {rmin}–{rmax} Ω·m *(_{desc}_)*")

        st.markdown("---")

        # 3) Grid resolution
        st.subheader("🔧 Resolusi Grid")
        nx = st.slider("Grid X", 20, 100, 60, 10)
        ny = st.slider("Grid Y", 10, 60, 30, 5)
        nz = st.slider("Grid Z", 10, 50, 25, 5)

        st.markdown("---")
        st.markdown("📖 *Referensi: Telford, W.M., Geldart, L.P., Sheriff, R.E. (1990). Applied Geophysics. Cambridge University Press.*")

    # ── Main content ──
    if uploaded is None:
        st.info("👆 Upload file Excel untuk memulai. Format: sheet **GEOMETRY** dan **DATA**.")
        # Show format guide
        with st.expander("📋 Panduan Format Data Excel"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sheet: GEOMETRY**")
                st.dataframe(pd.DataFrame({"Line": ["L1","L2","L3"], "Y_Position": [0, 10, 20]}))
            with c2:
                st.markdown("**Sheet: DATA**")
                st.dataframe(pd.DataFrame({
                    "Line":        ["L1","L1","L2"],
                    "Distance":    [1.5, 2.5, 1.5],
                    "Position":    [0,   0,   0],
                    "Depth":       [-0.25, -0.5, -0.25],
                    "Resistivity": [1.29,  2.44, 15.3],
                }))

        # Show Telford table reference
        with st.expander("📊 Tabel Resistivitas Telford (Referensi Lengkap)"):
            rows = []
            for name, (rmin, rmax, label, desc) in TELFORD_TABLE.items():
                rows.append({"Litologi": label, "Min (Ω·m)": rmin, "Max (Ω·m)": rmax, "Keterangan": desc})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    # ── Load & process ──
    with st.spinner("Membaca dan memproses data..."):
        try:
            df, line_y = load_data(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

    if df.empty:
        st.error("Data kosong atau format kolom tidak dikenali.")
        return

    active_lithologies = env_info["lithologies"]
    vmin = float(df["RHO"].quantile(0.02))
    vmax = float(df["RHO"].quantile(0.98))

    # ── Data Summary ──
    st.success(f"✅ Data berhasil dimuat: **{len(df):,}** titik pengukuran dari **{df['LINE'].nunique()}** lintasan")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Titik", f"{len(df):,}")
    col2.metric("Jumlah Lintasan", df["LINE"].nunique())
    col3.metric("Min Resistivitas", f"{df['RHO'].min():.2f} Ω·m")
    col4.metric("Max Resistivitas", f"{df['RHO'].max():.2f} Ω·m")
    col5.metric("Rata-rata", f"{df['RHO'].mean():.2f} Ω·m")

    if custom_notes:
        st.info(f"📝 **Catatan Geologi:** {custom_notes}")

    # ── Build Grid ──
    with st.spinner("Membangun grid 3D..."):
        xi, yi, zi, grid = build_grid(df, nx=nx, ny=ny, nz=nz)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧊 Model 3D Volume",
        "🏗️ Fence Diagram",
        "📐 Irisan Horizontal",
        "📏 Irisan Vertikal",
        "📊 Interpretasi Litologi",
    ])

    # ── Tab 1: 3D Volume ──
    with tab1:
        st.subheader("Model 3D Resistivitas")
        fig3d = plot_3d_volume(xi, yi, zi, grid, vmin, vmax,
                               title_suffix=f" – {env_choice}")
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            "Model volume 3D hasil interpolasi griddata (scipy). "
            "Warna merah = resistivitas rendah (konduktif), hijau = tinggi (resistif). "
            "Nilai ditampilkan dalam Ω·m (apparent resistivity)."
        )

    # ── Tab 2: Fence Diagram ──
    with tab2:
        st.subheader("Fence Diagram")
        fig_fence = plot_fence_diagram(df, line_y, vmin, vmax)
        st.plotly_chart(fig_fence, use_container_width=True)
        st.caption(
            "Penampang vertikal setiap lintasan ditampilkan dalam ruang 3D. "
            "Nilai resistivitas dalam Ω·m."
        )

    # ── Tab 3: Horizontal Slice ──
    with tab3:
        st.subheader("Irisan Horizontal")
        depth_labels = [f"{z:.2f} m" for z in zi]
        depth_label_sel = st.select_slider(
            "Pilih Kedalaman",
            options=depth_labels,
            value=depth_labels[len(depth_labels) // 2],
        )
        depth_idx = depth_labels.index(depth_label_sel)
        fig_hz = plot_horizontal_slice(xi, yi, zi, grid, depth_idx, zi[depth_idx], vmin, vmax)
        st.plotly_chart(fig_hz, use_container_width=True)

        # Dominant lithology at this depth
        slice_vals = grid[:, :, depth_idx].flatten()
        slice_vals = slice_vals[~np.isnan(slice_vals)]
        if len(slice_vals) > 0:
            median_val = float(np.median(slice_vals))
            primary = get_primary_lithology(median_val, active_lithologies)
            st.info(
                f"🔍 Kedalaman **{zi[depth_idx]:.2f} m** | "
                f"Median resistivitas: **{median_val:.2f} Ω·m** | "
                f"Interpretasi dominan: **{primary}**"
            )

    # ── Tab 4: Vertical Slice ──
    with tab4:
        st.subheader("Irisan Vertikal per Lintasan")
        line_labels = [f"Y = {y:.1f} m" for y in yi]
        line_label_sel = st.select_slider(
            "Pilih Lintasan (Y)",
            options=line_labels,
            value=line_labels[len(line_labels) // 2],
        )
        line_idx = line_labels.index(line_label_sel)
        fig_vt = plot_vertical_slice(xi, zi, grid, line_idx, yi[line_idx], vmin, vmax)
        st.plotly_chart(fig_vt, use_container_width=True)

    # ── Tab 5: Interpretasi Litologi ──
    with tab5:
        st.subheader(f"Interpretasi Litologi – {env_choice}")
        st.markdown(
            f"Interpretasi berdasarkan **Tabel Telford et al. (1990)** "
            f"dengan filter lingkungan geologi: **{env_choice}**."
        )

        # Lithology legend
        st.markdown("#### 📋 Tabel Referensi Litologi Aktif")
        legend_df = build_lithology_legend(active_lithologies, vmin, vmax)
        if not legend_df.empty:
            st.dataframe(legend_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Statistics
        st.markdown("#### 📊 Distribusi Litologi dalam Data")
        stats_df = build_stats_table(df, active_lithologies)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Bar chart distribusi
            fig_bar = go.Figure(go.Bar(
                x=stats_df["Litologi"],
                y=[float(v.replace("%","")) for v in stats_df["% Data"]],
                marker_color="steelblue",
                text=stats_df["% Data"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Distribusi Litologi (% titik data)",
                xaxis_title="Litologi",
                yaxis_title="% Data",
                height=400,
                margin=dict(l=0, r=0, t=40, b=120),
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Tidak ada litologi yang cocok dengan rentang data dalam lingkungan geologi ini.")

        st.markdown("---")

        # Point-wise interpretation
        st.markdown("#### 🔎 Interpretasi Titik Data")
        n_sample = min(200, len(df))
        sample_df = df.sample(n_sample, random_state=42).copy()
        sample_df["Litologi Utama"] = sample_df["RHO"].apply(
            lambda r: get_primary_lithology(r, active_lithologies)
        )
        display_cols = ["LINE", "X", "Y", "Z", "RHO", "Litologi Utama"]
        available_cols = [c for c in display_cols if c in sample_df.columns]
        rename_map = {"LINE": "Lintasan", "X": "Jarak (m)", "Y": "Y (m)",
                      "Z": "Kedalaman (m)", "RHO": "Resistivitas (Ω·m)"}
        st.dataframe(
            sample_df[available_cols].rename(columns=rename_map).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Menampilkan sampel {n_sample} titik dari total {len(df):,} titik.")

        st.markdown("---")
        st.markdown(
            "📖 **Referensi:** Telford, W.M., Geldart, L.P., & Sheriff, R.E. (1990). "
            "*Applied Geophysics* (2nd ed.). Cambridge University Press."
        )


if __name__ == "__main__":
    main()
