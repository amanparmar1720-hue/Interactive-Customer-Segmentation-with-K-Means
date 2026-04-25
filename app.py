import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Mall Customer Segmentation (KMeans)", layout="wide")


def pick_k_balanced(ks, sils, ratio=0.95):
    best = max(sils)
    threshold = ratio * best
    for k, s in zip(ks, sils):
        if s >= threshold:
            return k
    return ks[int(np.argmax(sils))]


def segment_label(income, spend):
    if income >= 70 and spend >= 70:
        return "VIP / Premium"
    if income >= 70 and spend <= 35:
        return "Potential (Upsell)"
    if income <= 40 and spend >= 70:
        return "Deal Seekers"
    if income <= 40 and spend <= 35:
        return "Low Value"
    return "Mainstream"


@st.cache_data(show_spinner=False)
def compute_scores(X_proc, k_min, k_max):
    ks = list(range(k_min, k_max + 1))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_proc)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_proc, labels))
    return ks, inertias, sils


@st.cache_data(show_spinner=False)
def fit_kmeans(X_proc, k):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_proc)
    return km, labels


st.title("🛍️ Mall Customer Segmentation (Interactive KMeans)")

st.sidebar.header("Settings")

use_default = st.sidebar.checkbox("Use default dataset (fixed)", value=True)

if use_default:
    try:
        df = pd.read_csv("Mall_Customers.csv")
        st.caption("Using built-in dataset: Mall_Customers.csv")
    except Exception:
        st.error("Mall_Customers.csv not found next to app.py. Either add it there or uncheck 'Use default dataset'.")
        st.stop()
else:
    uploaded = st.file_uploader("Upload CSV (Mall Customers)", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to start.")
        st.stop()
    df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head())

mode = st.sidebar.radio(
    "Clustering mode",
    ["Classic (Income + Spending) — recommended", "Multivariate (selected features)"]
)

k_min, k_max = st.sidebar.slider("k range", 2, 12, (2, 10))
auto_k = st.sidebar.checkbox("Auto-pick k (balanced)", value=True)

scale = st.sidebar.checkbox("Scale features (recommended for multivariate)", value=True)

features_default = [c for c in ["Annual Income (k$)", "Spending Score (1-100)", "Age"] if c in df.columns]
numeric_candidates = [c for c in df.columns if df[c].dtype != "object" and c.lower() != "cluster"]

features = features_default
include_gender = False

if mode.startswith("Multivariate"):
    features = st.sidebar.multiselect(
        "Select features for clustering",
        options=[c for c in numeric_candidates if c in df.columns],
        default=[c for c in features_default if c in numeric_candidates]
    )
    include_gender = st.sidebar.checkbox("Include Gender (encode Male=1, Female=0)", value=("Gender" in df.columns))
    if len(features) < 2:
        st.warning("Select at least 2 numeric features for multivariate clustering.")
        st.stop()

if "run" not in st.session_state:
    st.session_state.run = False

if st.sidebar.button("▶ Start / Re-run clustering"):
    st.session_state.run = True

if not st.session_state.run:
    st.info("Set options on the left, then click **Start / Re-run clustering** to compute elbow/silhouette and clusters.")
    st.stop()

data = df.copy()

if mode.startswith("Classic"):
    if "Annual Income (k$)" not in data.columns or "Spending Score (1-100)" not in data.columns:
        st.error("Classic mode needs 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")
        st.stop()
    features_used = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = data[features_used].values
    X_proc = X
else:
    X_cols = list(features)
    if include_gender and "Gender" in data.columns:
        data["Gender_Male"] = (data["Gender"].astype(str).str.lower() == "male").astype(int)
        X_cols.append("Gender_Male")

    X = data[X_cols].values

    if scale:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

st.spinner("Computing elbow & silhouette...")
ks, inertias, sils = compute_scores(X_proc, k_min, k_max)

best_k_by_sil = ks[int(np.argmax(sils))]
k_balanced = pick_k_balanced(ks, sils, ratio=0.95)

if auto_k:
    k_selected = 5 if mode.startswith("Classic") and 5 in ks else k_balanced
else:
    default_k = 5 if mode.startswith("Classic") and 5 in ks else k_balanced
    k_selected = st.sidebar.selectbox("Select k", ks, index=ks.index(default_k))

st.success(f"Selected k = {k_selected} | best silhouette in range = {best_k_by_sil} | balanced pick = {k_balanced}")

c1, c2 = st.columns(2)
with c1:
    fig = plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    st.pyplot(fig)

with c2:
    fig = plt.figure()
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    st.pyplot(fig)

km, labels = fit_kmeans(X_proc, k_selected)
data["cluster"] = labels

st.subheader("Cluster Profiling (means)")

numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

if "cluster" in numeric_cols:
    numeric_cols.remove("cluster")

profile = data.groupby("cluster")[numeric_cols].mean(numeric_only=True).round(2)
profile["count"] = data.groupby("cluster").size()
profile = profile.sort_values("count", ascending=False)

st.dataframe(profile)

if mode.startswith("Classic"):
    centers = pd.DataFrame(km.cluster_centers_, columns=features_used)
    centers["segment"] = centers.apply(lambda r: segment_label(r[features_used[0]], r[features_used[1]]), axis=1)
    st.subheader("Segment labels (from cluster centers)")
    st.dataframe(centers[features_used + ["segment"]])

if "Gender" in data.columns:
    st.subheader("Gender mix by cluster")
    mix = pd.crosstab(data["cluster"], data["Gender"], normalize="index").round(3)
    st.dataframe(mix)

st.subheader("Cluster visualization")
num_vis = data.select_dtypes(include=["number"]).columns.tolist()
num_vis = [c for c in num_vis if c not in ["cluster", "CustomerID"]]

if mode.startswith("Classic"):
    x_feat, y_feat = features_used[0], features_used[1]
else:
    x_feat = st.selectbox("X axis", num_vis, index=num_vis.index(features[0]) if features and features[0] in num_vis else 0)
    y_feat = st.selectbox("Y axis", num_vis, index=num_vis.index(features[1]) if len(features) > 1 and features[1] in num_vis else (1 if len(num_vis) > 1 else 0))

fig = plt.figure(figsize=(7, 5))
sns.scatterplot(data=data, x=x_feat, y=y_feat, hue="cluster", palette="tab10")
plt.title("Clusters")

if mode.startswith("Classic"):
    plt.scatter(centers[x_feat], centers[y_feat], s=200, marker="*", c="black")

st.pyplot(fig)

st.subheader("Download clustered data")
out_df = data.copy()
csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV with clusters", csv, "clustered_customers.csv", "text/csv")
