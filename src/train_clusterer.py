"""K-Means clustering on a curated behavioral feature subset."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import (
    CLUSTER_VIZ_PNG,
    ELBOW_PLOT_PNG,
    KMEANS_FEATURES,
    KMEANS_MODEL_PATH,
    KMEANS_PARAMS,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
    SCALER_PATH,
)
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


def _elbow_diagnostics(X_scaled: np.ndarray) -> dict:
    """Compute inertia + silhouette for k=2..8 and save plot."""
    ks = list(range(2, 9))
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertias, "o-", color="tab:blue", label="Inertia")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, "s-", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="tab:orange")
    ax1.set_title("KMeans elbow & silhouette")
    fig.tight_layout()
    fig.savefig(ELBOW_PLOT_PNG, dpi=120)
    plt.close(fig)
    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}


CLUSTER_NAME_CANDIDATES = [
    "Engaged Payers", "At-Risk Users", "Active Explorers",
    "Steady Mid-Tier", "Disengaged Lite",
]


def _cluster_names(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign unique descriptive names ranked by each cluster's dominant trait.

    Uses scaled (z-score) centroids. Each cluster gets a single best name from a
    fixed pool; names are not re-used so the labels are always unique.
    """
    # Score each cluster on each candidate persona; higher = better match.
    persona_scores = {}
    for i in centroids.index:
        c = centroids.loc[i]
        persona_scores[i] = {
            "Engaged Payers":   c["n_payment_success"] + c["checkout_to_payment_rate"] + c["activity_trend"],
            "At-Risk Users":    c["last_event_day"] - c["activity_trend"],
            "Active Explorers": c["n_feature_click"] + c["n_login"] - c["n_payment_success"],
            "Steady Mid-Tier":  -abs(c["n_payment_success"]) - abs(c["last_event_day"]),
            "Disengaged Lite":  -c["n_login"] - c["n_feature_click"] - c["n_payment_success"],
        }

    used = set()
    names = {}
    # Greedy assignment: each round, assign the highest unmatched (cluster, persona) pair.
    remaining = set(centroids.index)
    while remaining:
        best_pair = None
        best_score = float("-inf")
        for i in remaining:
            for p, s in persona_scores[i].items():
                if p in used:
                    continue
                if s > best_score:
                    best_score = s
                    best_pair = (i, p)
        if best_pair is None:
            # All personas used; fall back to numeric label.
            for i in remaining:
                names[i] = f"Cluster {i}"
            break
        i, p = best_pair
        names[i] = p
        used.add(p)
        remaining.discard(i)
    return names


def _save_pca_viz(X_scaled: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cid in sorted(np.unique(labels)):
        m = labels == cid
        ax.scatter(pts[m, 0], pts[m, 1], label=names[cid], alpha=0.7, s=50)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Behavioral clusters (PCA projection)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CLUSTER_VIZ_PNG, dpi=120)
    plt.close(fig)


def train(features: pd.DataFrame) -> dict:
    X = features[KMEANS_FEATURES].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    diag = _elbow_diagnostics(X_scaled)
    log.info("Elbow diagnostics: silhouette@k=4 = %.3f",
             diag["silhouettes"][diag["ks"].index(4)])

    km = KMeans(**KMEANS_PARAMS)
    labels = km.fit_predict(X_scaled)
    log.info("Cluster sizes: %s", dict(pd.Series(labels).value_counts().sort_index()))

    centroids_scaled = pd.DataFrame(km.cluster_centers_, columns=KMEANS_FEATURES)
    names = _cluster_names(centroids_scaled)
    log.info("Cluster names: %s", names)

    _save_pca_viz(X_scaled, labels, names)

    ensure_dir(MODELS_DIR)
    joblib.dump({"model": km, "feature_names": KMEANS_FEATURES, "names": names}, KMEANS_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Saved KMeans and scaler to %s, %s", KMEANS_MODEL_PATH, SCALER_PATH)

    # Persist centroids in original (unscaled) feature space for README readability.
    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=KMEANS_FEATURES,
    )
    centroids_orig["cluster_name"] = [names[i] for i in range(N_CLUSTERS)]

    meta_path = ensure_dir(MODEL_METADATA_PATH.parent) / MODEL_METADATA_PATH.name
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing.update({
        "clusterer": {
            "n_clusters": N_CLUSTERS,
            "feature_names": KMEANS_FEATURES,
            "names": {str(k): v for k, v in names.items()},
            "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
            "silhouette_k4": float(diag["silhouettes"][diag["ks"].index(4)]),
            "centroids_original_space": centroids_orig.round(3).to_dict(orient="records"),
        }
    })
    meta_path.write_text(json.dumps(existing, indent=2))

    return {
        "labels": labels,
        "names": names,
        "scaler": scaler,
        "kmeans": km,
        "diagnostics": diag,
        "centroids_original_space": centroids_orig,
    }
