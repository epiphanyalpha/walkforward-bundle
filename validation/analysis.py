"""
Post-processing of an ensemble run: paths in, structure out.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_oos_paths_dataframe(oos_series, config_index):
    # Keep only series present in index and drop None
    valid_keys = [k for k in config_index if k in oos_series and isinstance(oos_series[k], pd.Series)]
    if not valid_keys:
        return pd.DataFrame(index=config_index)

    # Drop duplicate timestamps in each series, build union of dates
    all_dates = pd.DatetimeIndex([])
    cleaned = {}
    for k in valid_keys:
        s = oos_series[k]
        s = s[~s.index.duplicated(keep='first')]
        cleaned[k] = s
        all_dates = all_dates.union(s.index)
    all_dates = all_dates.sort_values()

    # Reindex each to full date set, fill missing with zero
    df = pd.DataFrame(
        {k: cleaned[k].reindex(all_dates, fill_value=0) for k in valid_keys}
    ).T

    # Re-order rows to match original config_index, fill missing configs with zero
    df = df.reindex(config_index, fill_value=0)
    df.index.name = config_index.name
    return df
    
def prepare_oos_paths_dataframe_2(oos_series: dict, config_index: list) -> pd.DataFrame:
    """
    Given:
      oos_series   - dict mapping config_key -> pd.Series of OOS returns
      config_index - list of config_key in the order you want
    
    Returns a DataFrame with
      rows = config_key,
      columns = **all** timestamps seen by any series (union),
      values = return at that timestamp or NaN if missing.

    Differs from :func:`prepare_oos_paths_dataframe` in one respect that
    matters: gaps stay ``NaN`` instead of becoming ``0.0``. Use this one when
    you need to tell "flat" apart from "not trading yet" — clustering and
    equity plots generally want the zero-filled version instead.
    """
    # 1) Filter for keys present in both your dict and index
    valid_keys = [k for k in config_index if k in oos_series and oos_series[k] is not None]
    if not valid_keys:
        # nothing to show — return empty frame with correct index
        return pd.DataFrame(index=config_index)

    # 2) Build a dict of reindexed series over the UNION of all dates
    #    First compute the full union of all indexes
    all_dates = pd.Index([])
    for k in valid_keys:
        all_dates = all_dates.union(oos_series[k].index)
    all_dates = all_dates.sort_values()

    # 3) Reindex each series to that full date set
    reindexed = {
        k: oos_series[k].reindex(all_dates)
        for k in valid_keys
    }

    # 4) Build DataFrame (rows = configs, cols = dates)
    paths_df = pd.DataFrame(reindexed).T

    # 5) Finally, re‐order rows to match your original config_index
    return paths_df.reindex(config_index)





def calculate_clusters(oos_paths_df: pd.DataFrame,
                       results_df: pd.DataFrame,
                       n_clusters: int,
                       random_state: int = None) -> dict:
    """
    Cluster the out-of-sample paths (rows = configs, cols = timepoints)
    and attach the resulting cluster labels back onto results_df.
    Returns a dict with:
      - kmeans_model
      - scaler
      - cluster_labels (np.ndarray)
      - cluster_counts (np.ndarray)
      - results_with_clusters (DataFrame)
    """

    # NOTE: the paths are standardized before clustering, which deliberately
    # discards level and scale. Clusters therefore group configurations by the
    # *shape* of their out-of-sample path -- when they made and lost money --
    # not by how much. That is usually what you want from this: two configs
    # with the same shape are the same bet at different size.

    # 1. Scale each config-path to mean=0, std=1 across time
    scaler = StandardScaler()
    # Fill NaN with zero so scaler won't choke
    filled = oos_paths_df.fillna(0)
    scaled = scaler.fit_transform(filled)

    # 2. Fit KMeans and get labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(scaled)

    # 3. Build a results DataFrame aligned to exactly these configs
    #    Use reindex so that even if results_df has a different order or extra rows,
    #    we end up with one row per oos_paths_df.index, in the same order.
    results_with_clusters = results_df.reindex(oos_paths_df.index).copy()

    # 4. Turn labels into a Series indexed by the same configs
    labels_s = pd.Series(labels, index=oos_paths_df.index, name="cluster")

    # 5. Assign by index (pandas lines them up correctly)
    results_with_clusters["cluster"] = labels_s

    return {
        "kmeans_model":      kmeans,
        "scaler":            scaler,
        "cluster_labels":    labels,
        "cluster_counts":    np.bincount(labels),
        "results_with_clusters": results_with_clusters
    }
