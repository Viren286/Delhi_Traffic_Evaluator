import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Route Planner | Shortest Path Explorer",
    page_icon="🧭",
    layout="wide",
)

# =====================================================
# AESTHETIC — Blue background, white text, cyan accents
# =====================================================
st.markdown(
    """
    <style>
    :root{
        --bg-deep:#081b3f;
        --bg-panel:#0e2a5e;
        --bg-panel-2:#123268;
        --cyan:#22e5ff;
        --cyan-soft:#7be9ff;
        --white:#f4f9ff;
    }

    .stApp{
        background: radial-gradient(circle at top left, #10306b 0%, #081b3f 55%, #050f28 100%);
        color: var(--white);
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div{
        color: var(--white);
    }

    /* Titles */
    h1{
        color: var(--cyan) !important;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    h2, h3{
        color: var(--cyan-soft) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #0b234f 0%, #071a3c 100%);
        border-right: 1px solid rgba(34,229,255,0.25);
    }

    /* Cards / containers */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        background: var(--bg-panel);
        border: 1px solid rgba(34,229,255,0.25);
        border-radius: 14px;
        padding: 6px;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button{
        background: linear-gradient(135deg, #22e5ff, #0ea5c4);
        color: #04182f;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        box-shadow: 0 0 14px rgba(34,229,255,0.45);
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover{
        transform: translateY(-1px);
        box-shadow: 0 0 22px rgba(34,229,255,0.75);
        color: #04182f;
    }

    /* Selectboxes / inputs */
    div[data-baseweb="select"] > div{
        background-color: var(--bg-panel-2);
        border: 1px solid rgba(34,229,255,0.35);
        color: var(--white);
        border-radius: 8px;
    }
    input, textarea{
        background-color: var(--bg-panel-2) !important;
        color: var(--white) !important;
    }

    /* DataFrames / tables */
    .stDataFrame{
        border: 1px solid rgba(34,229,255,0.25);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Metric boxes */
    div[data-testid="stMetric"]{
        background: var(--bg-panel-2);
        border: 1px solid rgba(34,229,255,0.3);
        border-radius: 12px;
        padding: 12px;
    }

    /* Alerts */
    div[data-testid="stAlert"]{
        background: var(--bg-panel-2);
        border-radius: 10px;
    }

    /* Expander */
    details{
        background: var(--bg-panel-2);
        border-radius: 10px;
        border: 1px solid rgba(34,229,255,0.25);
    }

    hr{
        border-color: rgba(34,229,255,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Read CSV
# =====================================================
@st.cache_data(show_spinner=False)
def load_data(source):
    df = pd.read_csv(source)
    return df


# =====================================================
# Calculate trip time + build averaged dataframe
# =====================================================
@st.cache_data(show_spinner=False)
def build_avg_df(df: pd.DataFrame):
    df = df.copy()
    df["trip_time_minutes"] = (
        df["distance_km"] / df["average_speed_kmph"]
    ) * 60

    avg_df = (
        df.groupby(
            [
                "start_area",
                "end_area",
                "day_of_week",
                "weather_condition",
                "traffic_density_level",
                "time_of_day",
            ],
            as_index=False,
        )
        .agg(avg_time=("trip_time_minutes", "mean"))
    )
    return avg_df


# =====================================================
# Assign integer IDs to every area
# =====================================================
@st.cache_data(show_spinner=False)
def build_area_ids(avg_df: pd.DataFrame):
    areas = sorted(
        set(avg_df["start_area"]).union(set(avg_df["end_area"]))
    )
    area_to_id = {area: idx for idx, area in enumerate(areas)}
    id_to_area = {idx: area for area, idx in area_to_id.items()}
    return area_to_id, id_to_area


# =====================================================
# Create graphs (one per condition combination)
# =====================================================
GROUP_COLUMNS = [
    "day_of_week",
    "traffic_density_level",
    "time_of_day",
    "weather_condition",
]


@st.cache_resource(show_spinner=False)
def build_graphs(avg_df: pd.DataFrame, area_to_id: dict):
    graphs = {}
    for key, group in avg_df.groupby(GROUP_COLUMNS):
        G = nx.DiGraph()
        # Add all vertices
        for area, idx in area_to_id.items():
            G.add_node(idx, name=area)
        # Add weighted edges
        for _, row in group.iterrows():
            u = area_to_id[row["start_area"]]
            v = area_to_id[row["end_area"]]
            G.add_edge(u, v, weight=row["avg_time"])
        graphs[key] = G
    return graphs


# =====================================================
# Helpers for the frontend
# =====================================================
def path_total_time(G, path):
    return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


def path_label(path, id_to_area):
    return " → ".join(id_to_area[n] for n in path)


def find_best_available_graph(graphs, source, target):
    """
    Scan every condition-graph for one that actually contains a path
    between source and target. Returns the graph/key whose shortest
    path is shortest overall, so there is always a route to show even
    if the exact selected conditions have no data or no connectivity.
    Returns (key, G, shortest_path, shortest_time) or (None, None, None, None)
    if no graph connects the two areas at all.
    """
    best = None  # (time, key, G, path)
    for key, G in graphs.items():
        if source not in G or target not in G:
            continue
        try:
            p = nx.shortest_path(G, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath:
            continue
        t = path_total_time(G, p)
        if best is None or t < best[0]:
            best = (t, key, G, p)
    if best is None:
        return None, None, None, None
    _, key, G, p = best
    return key, G, p, path_total_time(G, p)


def find_all_paths(G, source, target, max_paths=25, cutoff=None):
    """Return up to max_paths simple paths, sorted by total time."""
    paths = []
    try:
        for p in nx.all_simple_paths(G, source, target, cutoff=cutoff):
            paths.append((p, path_total_time(G, p)))
            if len(paths) >= 2000:  # safety cap on search itself
                break
    except nx.NodeNotFound:
        return []
    paths.sort(key=lambda x: x[1])
    return paths[:max_paths]


CYAN_PALETTE = [
    "#22e5ff", "#7be9ff", "#00b8d9", "#5ac8fa", "#0ea5c4",
    "#8fd6ff", "#2ec4f1", "#4fd0e0", "#1fb6d6", "#6fe3ff",
    "#0891b2", "#38bdf8", "#67e8f9", "#0e7490", "#a5f3fc",
]


def draw_pie(G, path, id_to_area, title):
    segments = []
    weights = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        w = G[u][v]["weight"]
        segments.append(f"{id_to_area[u]} → {id_to_area[v]}")
        weights.append(w)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    fig.patch.set_facecolor("#081b3f")
    ax.set_facecolor("#081b3f")

    colors = [CYAN_PALETTE[i % len(CYAN_PALETTE)] for i in range(len(weights))]

    wedges, texts, autotexts = ax.pie(
        weights,
        labels=segments,
        autopct=lambda pct: f"{pct:.1f}%\n({pct/100*sum(weights):.1f} min)",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "#081b3f", "linewidth": 2},
        textprops={"color": "white", "fontsize": 9},
        pctdistance=0.72,
    )
    for t in autotexts:
        t.set_color("#04182f")
        t.set_fontweight("bold")
        t.set_fontsize(8)

    ax.set_title(title, color="#22e5ff", fontsize=13, fontweight="bold", pad=16)
    ax.axis("equal")
    fig.tight_layout()
    return fig


# =====================================================
# APP HEADER
# =====================================================
st.title("🧭 Smart Route Explorer")
st.caption(
    "Condition-aware shortest path finder — routes adapt to day, traffic, "
    "time of day and weather, built on the averaged-travel-time graph model."
)

# =====================================================
# DATA SOURCE
# =====================================================
default_path = "trips.csv"
df = None

if os.path.exists(default_path):
    df = load_data(default_path)
else:
    st.info("No `trips.csv` found next to the app — upload your dataset to begin.")
    uploaded = st.file_uploader("Upload trips.csv", type=["csv"])
    if uploaded is not None:
        df = load_data(uploaded)

if df is None:
    st.stop()

required_cols = {
    "distance_km", "average_speed_kmph", "start_area", "end_area",
    "day_of_week", "weather_condition", "traffic_density_level", "time_of_day",
}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"The CSV is missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

avg_df = build_avg_df(df)
area_to_id, id_to_area = build_area_ids(avg_df)
graphs = build_graphs(avg_df, area_to_id)

# =====================================================
# SIDEBAR — trip conditions
# =====================================================
with st.sidebar:
    st.header("⚙️ Trip Conditions")

    areas_sorted = sorted(area_to_id.keys())
    start_area = st.selectbox("Start Area", areas_sorted, index=0)
    end_options = [a for a in areas_sorted if a != start_area]
    end_area = st.selectbox("End Area", end_options, index=0 if end_options else None)

    st.markdown("---")

    day_of_week = st.selectbox("Day of Week", sorted(avg_df["day_of_week"].unique()))
    time_of_day = st.selectbox("Time of Day", sorted(avg_df["time_of_day"].unique()))
    weather_condition = st.selectbox(
        "Weather Condition", sorted(avg_df["weather_condition"].unique())
    )
    traffic_density_level = st.selectbox(
        "Traffic Density", sorted(avg_df["traffic_density_level"].unique())
    )

    st.markdown("---")
    max_paths_to_show = st.slider("Max alternative paths to list", 1, 50, 15)
    find_clicked = st.button("🔍 Find Routes", use_container_width=True)

st.markdown(
    f"**Graph vertices:** {len(area_to_id)} areas &nbsp;|&nbsp; "
    f"**Condition graphs built:** {len(graphs)}"
)

# =====================================================
# MAIN — results
# =====================================================
if "results" not in st.session_state:
    st.session_state.results = None

if find_clicked:
    key = (day_of_week, traffic_density_level, time_of_day, weather_condition)
    u = area_to_id[start_area]
    v = area_to_id[end_area]

    fallback_used = False
    used_key = key
    G = graphs.get(key)
    shortest, shortest_time = None, None

    if G is not None:
        try:
            shortest = nx.shortest_path(G, source=u, target=v, weight="weight")
            shortest_time = path_total_time(G, shortest)
        except nx.NetworkXNoPath:
            shortest = None

    # No graph for these exact conditions, or no path in that graph
    # -> fall back to whichever available graph connects the two areas.
    if shortest is None:
        fb_key, fb_G, fb_path, fb_time = find_best_available_graph(graphs, u, v)
        if fb_G is not None:
            fallback_used = True
            used_key = fb_key
            G = fb_G
            shortest = fb_path
            shortest_time = fb_time

    if G is None or shortest is None:
        st.error(
            "These two areas are not connected in any available condition graph — "
            "there's no trip data linking them at all."
        )
        st.session_state.results = None
    else:
        all_paths = find_all_paths(G, u, v, max_paths=max_paths_to_show)

        st.session_state.results = {
            "key": used_key,
            "requested_key": key,
            "fallback_used": fallback_used,
            "G": G,
            "shortest": shortest,
            "shortest_time": shortest_time,
            "all_paths": all_paths,
            "start_area": start_area,
            "end_area": end_area,
        }

results = st.session_state.results

if results:
    G = results["G"]

    st.subheader(f"Route: {results['start_area']} → {results['end_area']}")

    if results.get("fallback_used"):
        req = results["requested_key"]
        st.warning(
            f"No route existed under your selected conditions "
            f"(Day: **{req[0]}**, Traffic: **{req[1]}**, Time of day: **{req[2]}**, "
            f"Weather: **{req[3]}**). Showing the closest available route instead, "
            f"found under different conditions below."
        )

    st.caption(
        f"Conditions used → Day: **{results['key'][0]}**, Traffic: **{results['key'][1]}**, "
        f"Time of day: **{results['key'][2]}**, Weather: **{results['key'][3]}**"
    )

    if results["shortest"] is None:
        st.warning("No path exists between these two areas under the selected conditions.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Shortest travel time", f"{results['shortest_time']:.2f} min")
        with col2:
            st.success(path_label(results["shortest"], id_to_area))

        st.markdown("### 🛣️ All Possible Paths")
        if not results["all_paths"]:
            st.info("Only the shortest path is available between these areas.")
        else:
            table_rows = [
                {
                    "Rank": i + 1,
                    "Route": path_label(p, id_to_area),
                    "Total Time (min)": round(t, 2),
                }
                for i, (p, t) in enumerate(results["all_paths"])
            ]
            st.dataframe(
                pd.DataFrame(table_rows), use_container_width=True, hide_index=True
            )

            st.markdown("### 🥧 Visualize a Path's Time Breakdown")
            path_choices = {
                f"#{i+1} — {path_label(p, id_to_area)}  ({t:.2f} min)": p
                for i, (p, t) in enumerate(results["all_paths"])
            }
            chosen_label = st.selectbox("Select a path to view its pie chart", list(path_choices.keys()))
            chosen_path = path_choices[chosen_label]

            if len(chosen_path) < 2:
                st.info("This path has no segments to chart.")
            else:
                fig = draw_pie(
                    G,
                    chosen_path,
                    id_to_area,
                    title=f"Time share by segment — {path_label(chosen_path, id_to_area)}",
                )
                st.pyplot(fig, use_container_width=False)
else:
    st.info("Set your trip conditions in the sidebar and click **Find Routes** to begin.")
