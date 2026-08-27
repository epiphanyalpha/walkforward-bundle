"""
Plotting helpers for an ensemble run. Requires the ``[analysis]`` extra.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# --- Grouped Equity Curves ---
def plot_grouped_equity_curves(results_df, oos_series_dict, grouping_column, show_raw_curves=False):
    valid_keys = results_df.index.intersection(oos_series_dict.keys())
    results_df = results_df.loc[valid_keys]
    if results_df.empty:
        return
    grouped = results_df.groupby(grouping_column)
    for group, group_df in grouped:
        series_list = [oos_series_dict[k].fillna(0) for k in group_df.index if k in oos_series_dict]
        if not series_list:
            continue
        common_index = series_list[0].index
        df = pd.concat([s.reindex(common_index).fillna(0) for s in series_list], axis=1)
        cum = (1 + df).cumprod(axis=0) - 1
        mean_curve = cum.mean(axis=1)
        lower = cum.quantile(0.25, axis=1)
        upper = cum.quantile(0.75, axis=1)
        plt.figure(figsize=(10, 6))
        if show_raw_curves:
            for col in cum.columns:
                plt.plot(cum.index, cum[col], color='gray', alpha=0.2, linewidth=0.8)
        plt.plot(mean_curve, label=f"{group} Mean", color='blue', linewidth=2)
        plt.fill_between(mean_curve.index, lower, upper, alpha=0.2, color='blue')
        plt.title(f"Grouped Equity Curve by {grouping_column} = {group}")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Time")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Cluster Centroids ---
def plot_cluster_centroids(centroids, time_index, cluster_counts, scaler=None, cumulative=False):
    if scaler:
        centroids = scaler.inverse_transform(centroids)
    df = pd.DataFrame(centroids, columns=time_index)
    if cumulative:
        df = (1 + df).cumprod(axis=1) - 1
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        label = f"Cluster {i} (n={cluster_counts.get(i)})"
        plt.plot(time_index, row, label=label)
    plt.title("Cluster Centroids")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return" if cumulative else "Return")
    if cumulative:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Sharpe Boxplots ---
# Elegant boxplot with strip overlay and median annotations
def plot_grouped_performance_boxplots(
    results_df,
    group_by_columns,
    metric_column='oos_sharpe',
    palette="viridis"
):
    if metric_column not in results_df.columns:
        print(f"Skipping boxplots: '{metric_column}' not found.")
        return

    for col in group_by_columns:
        if col not in results_df.columns:
            continue

        #group_medians = results_df.groupby(col)[metric_column].median().sort_values()
        #order = group_medians.index.tolist()
        order = sorted(results_df[col].unique())


        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=results_df,
            x=col,
            y=metric_column,
            order=order,
            palette=palette,
            fliersize=0,
            width=0.6
        )

        sns.stripplot(
            data=results_df,
            x=col,
            y=metric_column,
            order=order,
            color='black',
            alpha=0.3,
            jitter=0.15,
            size=3
        )

        # Annotate median values (optional, now computed on demand)
        for i, val in enumerate(order):
            median_val = results_df[results_df[col] == val][metric_column].median()
            plt.text(i, median_val, f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, color='darkblue')

        plt.title(f"{metric_column} distribution as function of {col}", fontsize=14)
        plt.ylabel(metric_column)
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- PCA Config Space ---
def plot_config_space_pca(results_df, features, metric_column='oos_sharpe', n_components=2, highlight_top=5):
    df = results_df.dropna(subset=[metric_column])
    df_enc = pd.get_dummies(df[features], drop_first=True)
    X = StandardScaler().fit_transform(df_enc)
    pc = PCA(n_components=n_components)
    comps = pc.fit_transform(X)
    proj = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)
    proj[metric_column] = df[metric_column]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj['PC1'], proj['PC2'], c=proj[metric_column], cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(sc, label=metric_column)
    if highlight_top:
        top = proj.nlargest(highlight_top, metric_column)
        plt.scatter(top['PC1'], top['PC2'], color='red', marker='*', s=100, label='Top configs')
        plt.legend()
    plt.title("PCA of Config Space Colored by Sharpe")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Sharpe Heatmap ---
def plot_sharpe_heatmap(
    results_df,
    row_param,
    col_param,
    metric_column='oos_sharpe',
    cmap='crest',
    annotate=True,
    fmt=".2f",
    vmin=None,
    vmax=None
):
    pivot = results_df.pivot_table(
        index=row_param,
        columns=col_param,
        values=metric_column,
        aggfunc='mean'
    )

    # sort both axes when they are orderable
    try:
        pivot = pivot.sort_index().sort_index(axis=1)
    except TypeError:
        pass

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor='lightgray',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric_column}
    )
    plt.title(f"Mean {metric_column} by {row_param} vs {col_param}", fontsize=14)
    plt.xlabel(col_param)
    plt.ylabel(row_param)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- Cluster Summary & Details ---
def print_top_clusters(results_df, cluster_column='cluster', metric_column='oos_sharpe'):
    if cluster_column not in results_df.columns:
        warnings.warn(f"'{cluster_column}' not in results_df.")
        return
    grp = results_df.groupby(cluster_column)
    summary = grp[metric_column].agg(['mean','median','count']).sort_values('mean', ascending=False)
    print("Top Clusters by OOS Sharpe:")
    print(summary)
    for cid, g in grp:
        print(f"\nCluster {cid} (size={len(g)}) sample configs:")
        print(
            g.sort_values(metric_column, ascending=False)
             .head()[['metric_name','window_length','step_months','max_corr','max_columns', metric_column]]
        )

# --- UMAP Config Space (Nonlinear) ---
def plot_config_space_umap(results_df, features, metric_column='oos_sharpe', n_neighbors=15, min_dist=0.1, metric='euclidean', highlight_top=5):
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")
    df = results_df.dropna(subset=[metric_column])
    df_enc = pd.get_dummies(df[features], drop_first=True)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding = reducer.fit_transform(StandardScaler().fit_transform(df_enc))
    proj = pd.DataFrame(embedding, columns=['UMAP1','UMAP2'], index=df.index)
    proj[metric_column] = df[metric_column]

    import plotly.express as px
    fig = px.scatter(
        proj, x='UMAP1', y='UMAP2', color=metric_column, hover_name=proj.index,
        title='UMAP of Config Space by Sharpe', color_continuous_scale='Viridis'
    )
    if highlight_top > 0:
        top_idx = proj[metric_column].nlargest(highlight_top).index
        fig.add_scatter(
            x=proj.loc[top_idx,'UMAP1'], y=proj.loc[top_idx,'UMAP2'],
            mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Top configs'
        )
    fig.show()
