#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


ID_COLUMNS = {
    "orig_id",
    "vendor_id",
    "session_id",
    "conversation_id",
    "audio_path",
    "speakers",
    "relationship",
    "relationship_detail",
    "question_end_time",
    "extraction_status",
}
STATUS_SUBSTRINGS = (
    "status",
    "source",
)
ID_OUTPUT_COLUMNS = ["orig_id", "audio_path", "conversation_id", "relationship", "relationship_detail"]


def feature_path(results_root: Path, model: str, split: str, subset: str) -> Path:
    return results_root / model / split / subset / "distrib_baselines_features.csv"


def values_output_path(results_root: Path, model: str, subset: str, threshold: float) -> Path:
    suffix = threshold_suffix(threshold)
    return results_root / model / "test" / subset / f"correlation_cluster_features_rho{suffix}.csv"


def groups_output_path(results_root: Path, model: str, subset: str, threshold: float) -> Path:
    suffix = threshold_suffix(threshold)
    return results_root / model / "test" / subset / f"correlation_feature_groups_rho{suffix}.csv"


def threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature CSV: {path}")
    return pd.read_csv(path)


def is_candidate_feature(column: str) -> bool:
    if column in ID_COLUMNS:
        return False
    lower = column.lower()
    return not any(part in lower for part in STATUS_SUBSTRINGS)


def numeric_feature_columns(frames: Iterable[pd.DataFrame], min_valid: int) -> List[str]:
    common: Optional[set[str]] = None
    frame_list = list(frames)
    for frame in frame_list:
        cols = set(frame.columns)
        common = cols if common is None else common & cols
    if not common:
        return []

    features: List[str] = []
    for column in sorted(common):
        if not is_candidate_feature(column):
            continue
        usable = True
        for frame in frame_list:
            values = finite_series(frame, column)
            if values.notna().sum() < min_valid or values.dropna().nunique() < 2:
                usable = False
                break
        if usable:
            features.append(column)
    return features


def finite_series(frame: pd.DataFrame, feature: str) -> pd.Series:
    return pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_with_dev(dev: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dev_values = dev[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test_values = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    means = dev_values.mean(axis=0, skipna=True)
    stds = dev_values.std(axis=0, skipna=True).replace(0, np.nan)
    keep = stds.dropna().index.tolist()
    if len(keep) != len(features):
        dev_values = dev_values[keep]
        test_values = test_values[keep]
        means = means[keep]
        stds = stds[keep]

    dev_norm = (dev_values - means) / stds
    test_norm = (test_values - means) / stds
    return dev_norm, test_norm, means, stds


def normalize_frame(frame: pd.DataFrame, features: List[str], means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    values = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return (values - means) / stds


def spearman_abs_corr(frame: pd.DataFrame) -> pd.DataFrame:
    corr = frame.corr(method="spearman", min_periods=3).abs()
    corr = corr.reindex(index=frame.columns, columns=frame.columns)
    corr = corr.fillna(0.0).clip(0.0, 1.0)
    corr_values = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)


def cluster_features(abs_corr: pd.DataFrame, threshold: float) -> List[List[str]]:
    features = list(abs_corr.columns)
    if not features:
        return []
    if len(features) == 1:
        return [features]

    distance = 1.0 - abs_corr.to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")

    groups = []
    for cluster_id in sorted(set(labels)):
        group = sorted(feature for feature, label in zip(features, labels) if label == cluster_id)
        groups.append(group)
    return sorted(groups, key=lambda group: (len(group) == 1, group[0]))


def pairwise_abs_values(abs_corr: pd.DataFrame, features: List[str]) -> np.ndarray:
    if len(features) < 2:
        return np.array([], dtype=float)
    values = []
    for idx, left in enumerate(features):
        for right in features[idx + 1:]:
            values.append(float(abs_corr.loc[left, right]))
    return np.asarray(values, dtype=float)


def fit_cluster_component(dev_norm: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, float]:
    if len(features) == 1:
        return np.array([1.0], dtype=float), 1.0

    dev_matrix = dev_norm[features].fillna(0.0).to_numpy(dtype=float)
    pca = PCA(n_components=1)
    pca.fit(dev_matrix)
    component = pca.components_[0]
    if np.nansum(component) < 0:
        component = -component
    explained = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) else np.nan
    return component, explained


def transform_cluster(frame_norm: pd.DataFrame, features: List[str], component: np.ndarray) -> np.ndarray:
    matrix = frame_norm[features].fillna(0.0).to_numpy(dtype=float)
    return matrix @ component


def fit_general_component(cluster_dev_values: pd.DataFrame) -> Tuple[np.ndarray, float]:
    if cluster_dev_values.shape[1] == 1:
        return np.array([1.0], dtype=float), 1.0
    matrix = cluster_dev_values.fillna(0.0).to_numpy(dtype=float)
    pca = PCA(n_components=1)
    pca.fit(matrix)
    component = pca.components_[0]
    if np.nansum(component) < 0:
        component = -component
    explained = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) else np.nan
    return component, explained


def analyze_subset(results_root: Path, model: str, subset: str, thresholds: List[float], min_valid: int) -> List[Path]:
    target_models = ["original"] if model == "original" else ["original", model]
    dev_frames = {target: read_features(feature_path(results_root, target, "dev", subset)) for target in target_models}
    test_frames = {target: read_features(feature_path(results_root, target, "test", subset)) for target in target_models}

    features = numeric_feature_columns([*dev_frames.values(), *test_frames.values()], min_valid=min_valid)
    if not features:
        raise ValueError(f"No usable numeric explainable features for model={model} subset={subset}")

    combined_dev = pd.concat(dev_frames.values(), ignore_index=True)
    combined_test = pd.concat(test_frames.values(), ignore_index=True)
    combined_dev_norm, combined_test_norm, means, stds = normalize_with_dev(combined_dev, combined_test, features)
    features = list(combined_dev_norm.columns)
    dev_norm_by_model = {target: normalize_frame(frame, features, means, stds) for target, frame in dev_frames.items()}
    test_norm_by_model = {target: normalize_frame(frame, features, means, stds) for target, frame in test_frames.items()}

    dev_abs_corr = spearman_abs_corr(combined_dev_norm)
    test_abs_corr = spearman_abs_corr(combined_test_norm)

    written = []
    for threshold in thresholds:
        groups = cluster_features(dev_abs_corr, threshold)
        cluster_specs = []
        combined_dev_cluster_values = pd.DataFrame(index=combined_dev_norm.index)
        base_group_rows = []

        for group_idx, group_features in enumerate(groups, start=1):
            col_name = f"corr_cluster_{group_idx:03d}_rho{threshold_suffix(threshold)}"
            component, explained = fit_cluster_component(combined_dev_norm, group_features)
            combined_dev_cluster_values[col_name] = transform_cluster(combined_dev_norm, group_features, component)
            cluster_specs.append((col_name, group_features, component, explained))

            dev_pairwise = pairwise_abs_values(dev_abs_corr, group_features)
            test_pairwise = pairwise_abs_values(test_abs_corr, group_features)
            if len(group_features) == 1:
                validates = True
                min_test_abs = np.nan
                mean_test_abs = np.nan
                min_dev_abs = np.nan
                mean_dev_abs = np.nan
            else:
                min_dev_abs = float(np.nanmin(dev_pairwise))
                mean_dev_abs = float(np.nanmean(dev_pairwise))
                min_test_abs = float(np.nanmin(test_pairwise))
                mean_test_abs = float(np.nanmean(test_pairwise))
                validates = bool(min_test_abs >= threshold)

            base_group_rows.append(
                {
                    "subset": subset,
                    "threshold_abs_rho": threshold,
                    "cluster_id": group_idx,
                    "cluster_feature": col_name,
                    "n_features": len(group_features),
                    "features": "|".join(group_features),
                    "dev_min_abs_spearman": min_dev_abs,
                    "dev_mean_abs_spearman": mean_dev_abs,
                    "test_min_abs_spearman": min_test_abs,
                    "test_mean_abs_spearman": mean_test_abs,
                    "validates_in_test": validates,
                    "pca_explained_variance_ratio": explained,
                    "pca_scope": "original+model_dev",
                }
            )

        general_col = f"general_explainable_feature_rho{threshold_suffix(threshold)}"
        general_component, general_explained = fit_general_component(combined_dev_cluster_values)

        for target in target_models:
            test = test_frames[target]
            values_out = pd.DataFrame()
            for column in ID_OUTPUT_COLUMNS:
                if column in test.columns:
                    values_out[column] = test[column]

            for col_name, group_features, component, _ in cluster_specs:
                values_out[col_name] = transform_cluster(test_norm_by_model[target], group_features, component)
            if cluster_specs:
                cluster_cols = [col_name for col_name, _, _, _ in cluster_specs]
                values_out[general_col] = values_out[cluster_cols].fillna(0.0).to_numpy(dtype=float) @ general_component

            group_rows = []
            for row in base_group_rows:
                row = dict(row)
                row["model"] = target
                group_rows.append(row)
            group_rows.append(
                {
                    "subset": subset,
                    "model": target,
                    "threshold_abs_rho": threshold,
                    "cluster_id": "general",
                    "cluster_feature": general_col,
                    "n_features": len(cluster_specs),
                    "features": "|".join(col_name for col_name, _, _, _ in cluster_specs),
                    "dev_min_abs_spearman": np.nan,
                    "dev_mean_abs_spearman": np.nan,
                    "test_min_abs_spearman": np.nan,
                    "test_mean_abs_spearman": np.nan,
                    "validates_in_test": True,
                    "pca_explained_variance_ratio": general_explained,
                    "pca_scope": "all_correlation_clusters_original+model_dev",
                }
            )

            groups_out = pd.DataFrame(group_rows)
            values_path = values_output_path(results_root, target, subset, threshold)
            groups_path = groups_output_path(results_root, target, subset, threshold)
            values_path.parent.mkdir(parents=True, exist_ok=True)
            values_out.to_csv(values_path, index=False)
            groups_out.to_csv(groups_path, index=False)
            written.extend([groups_path, values_path])
            print(f"Wrote feature groups: {groups_path}")
            print(f"Wrote test cluster features: {values_path}")

    return written

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", nargs="+", default=["improvised", "naturalistic"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--min-valid", type=int, default=5)
    parser.add_argument("--ignore-features", nargs="*", default=[])
    args = parser.parse_args()
    ID_COLUMNS.update(args.ignore_features)

    thresholds = sorted(set(args.thresholds))
    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("Correlation thresholds must be between 0 and 1.")

    for subset in args.subsets:
        print(f"Analyzing explainable-feature correlations for model={args.model} subset={subset}")
        analyze_subset(args.results_root, args.model, subset, thresholds, args.min_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
