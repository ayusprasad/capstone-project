"""
promote_model.py

Select the best registered model by a chosen metric and promote it to
Production, archiving lower-performing versions. Works with a remote
MLflow Model Registry when available, and falls back to a local
`reports/model_registry.json` file created by the pipeline when the
registry or DagsHub is unavailable.

Usage:
    python scripts/promote_model.py --model-name my_model --metric accuracy

By default this promotes the highest value of the metric. Use
`--minimize` if lower is better (e.g., loss).

This script is safe to run in CI: it will attempt remote operations
when possible but always write back an updated local registry JSON so
DVC downstream steps can rely on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

LOCAL_REGISTRY_PATH = "reports/model_registry.json"


def load_local_registry(path: str = LOCAL_REGISTRY_PATH) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.info("Local registry file not found at %s", path)
        return {}
    except Exception as e:
        logging.warning("Failed to load local registry: %s", e)
        return {}


def save_local_registry(reg: Dict, path: str = LOCAL_REGISTRY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    logging.info("Local registry saved to %s", path)


def choose_best_from_local(
    reg: Dict, model_name: str, metrics: List[str], weights: Optional[List[float]], minimize: bool
) -> Optional[Tuple[str, Dict]]:
    """Select best version from local registry using multiple metrics.

    - metrics: ordered list of metric keys to use (e.g. ['accuracy','auc','recall'])
    - weights: optional weights matching metrics; if omitted equal weights used
    - minimize: if True lower is better for all metrics (applies as a global flag)

    The function normalizes each metric across versions (min-max) and computes
    a weighted average score. Versions missing a metric are penalized (treated as 0).
    """
    model = reg.get("models", {}).get(model_name)
    if not model:
        logging.info("Model %s not found in local registry", model_name)
        return None

    versions = [v for v in model.get("versions", []) if v.get("version") is not None]
    if not versions:
        logging.info("No versions present in local registry for %s", model_name)
        return None

    # Prepare weights
    if not metrics:
        logging.info("No metrics provided for selection")
        return None
    if weights is None:
        weights = [1.0] * len(metrics)
    # normalize weights
    total_w = sum(weights) if sum(weights) > 0 else 1.0
    weights = [w / total_w for w in weights]

    # Collect metric values per metric key
    metric_values: List[List[Optional[float]]] = []
    for m in metrics:
        vals = []
        for v in versions:
            vals.append((v.get("metrics") or {}).get(m))
        metric_values.append(vals)

    # Compute min/max for each metric and normalize
    norm_values_per_metric: List[List[float]] = []
    for vals in metric_values:
        present = [x for x in vals if x is not None]
        if not present:
            # nobody has this metric; use zeros
            norm_values_per_metric.append([0.0 for _ in vals])
            continue
        vmin = min(present)
        vmax = max(present)
        if vmax == vmin:
            # all equal -> assign 1.0 for those with value, 0 for missing
            norm = [1.0 if x is not None else 0.0 for x in vals]
        else:
            norm = [((x - vmin) / (vmax - vmin)) if x is not None else 0.0 for x in vals]
        # if minimize flag is set, invert normalization
        if minimize:
            norm = [1.0 - n for n in norm]
        norm_values_per_metric.append(norm)

    # Compute weighted score per version
    scores: List[float] = []
    for i in range(len(versions)):
        score = 0.0
        for mi in range(len(metrics)):
            score += weights[mi] * norm_values_per_metric[mi][i]
        scores.append(score)

    # pick highest score
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_ver = versions[best_idx]
    best_score = scores[best_idx]
    info = {"version": best_ver.get("version"), "score": best_score, "metrics_used": metrics}
    return str(best_ver.get("version")), info


def try_remote_promote(model_name: str, promote_version: str, archive_versions: List[str]) -> bool:
    """Attempt to promote using MLflow remote registry. Return True on success.

    If any MLflow operation fails, return False so caller can fallback to local.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:
        logging.info("MLflow not available for remote promotion: %s", e)
        return False

    try:
        client = MlflowClient()

        # Transition the chosen version to Production
        try:
            # Promote chosen version and archive any existing Production versions
            client.transition_model_version_stage(
                name=model_name,
                version=promote_version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info("Remote: promoted %s v%s -> Production (archived existing)", model_name, promote_version)
        except Exception as e:
            logging.warning("Remote promotion failed for %s v%s: %s", model_name, promote_version, e)
            # continue to try archiving others

        # Archive other versions
        for v in archive_versions:
            if v == promote_version:
                continue
            try:
                client.transition_model_version_stage(name=model_name, version=v, stage="Archived")
                logging.info("Remote: archived %s v%s", model_name, v)
            except Exception as e:
                logging.warning("Remote archive failed for %s v%s: %s", model_name, v, e)

        return True
    except Exception as e:
        logging.warning("Remote registry operations failed: %s", e)
        return False


def update_local_registry_after_promotion(reg: Dict, model_name: str, promote_version: str) -> Dict:
    # Ensure structure
    if "models" not in reg:
        reg["models"] = {}
    if model_name not in reg["models"]:
        reg["models"][model_name] = {"versions": []}

    # Update versions list: mark chosen as Production and others Archived
    for v in reg["models"][model_name].get("versions", []):
        if str(v.get("version")) == str(promote_version):
            v["stage"] = "production"
            v["status"] = "production"
        else:
            v["stage"] = "archived"
            v["status"] = "archived"
    return reg


def get_all_versions_local(reg: Dict, model_name: str) -> List[str]:
    model = reg.get("models", {}).get(model_name)
    if not model:
        return []
    return [str(v.get("version")) for v in model.get("versions", []) if v.get("version") is not None]


def choose_best_remote(model_name: str, metrics: List[str], weights: Optional[List[float]], minimize: bool) -> Optional[Tuple[str, Dict]]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:
        logging.info("MLflow not available for remote selection: %s", e)
        return None

    client = MlflowClient()
    try:
        # search_model_versions returns ModelVersion objects; we convert to dicts
        versions = client.search_model_versions(f"name = '{model_name}'")
        # Build table of metric values per version similar to local logic
        candidate_versions = []
        metric_table: List[List[Optional[float]]] = []
        for mv in versions:
            run_id = getattr(mv, "run_id", None)
            if not run_id:
                continue
            try:
                run = client.get_run(run_id)
                run_metrics = run.data.metrics or {}
            except Exception:
                # skip runs we can't access
                continue
            candidate_versions.append(mv)
            metric_row = [run_metrics.get(m) for m in metrics]
            metric_table.append(metric_row)

        if not candidate_versions:
            return None

        # transpose metric_table to per-metric lists
        norm_values_per_metric: List[List[float]] = []
        for mi in range(len(metrics)):
            vals = [row[mi] for row in metric_table]
            present = [x for x in vals if x is not None]
            if not present:
                norm_values_per_metric.append([0.0 for _ in vals])
                continue
            vmin = min(present)
            vmax = max(present)
            if vmax == vmin:
                norm = [1.0 if x is not None else 0.0 for x in vals]
            else:
                norm = [((x - vmin) / (vmax - vmin)) if x is not None else 0.0 for x in vals]
            if minimize:
                norm = [1.0 - n for n in norm]
            norm_values_per_metric.append(norm)

        # weights handling
        if weights is None:
            weights = [1.0] * len(metrics)
        total_w = sum(weights) if sum(weights) > 0 else 1.0
        weights = [w / total_w for w in weights]

        # compute scores
        scores = []
        for i in range(len(candidate_versions)):
            s = 0.0
            for mi in range(len(metrics)):
                s += weights[mi] * norm_values_per_metric[mi][i]
            scores.append(s)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_mv = candidate_versions[best_idx]
        info = {"version": best_mv.version, "run_id": getattr(best_mv, "run_id", None), "score": scores[best_idx]}
        return str(best_mv.version), info
    except Exception as e:
        logging.warning("Failed to select best remotely: %s", e)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True, help="Model name in registry (e.g. my_model)")
    p.add_argument("--metric", default="accuracy", help="Metric to select best model by")
    p.add_argument("--minimize", action="store_true", help="If set, lower metric is better (e.g. loss)")
    p.add_argument("--local-only", action="store_true", help="Force local-only mode (don't call MLflow)")
    p.add_argument(
        "--metrics",
        default=None,
        help=("Comma-separated list of metrics to use for selection (e.g. accuracy,auc,recall). "
              "If omitted defaults to accuracy,auc,recall"),
    )
    p.add_argument(
        "--weights",
        default=None,
        help=("Comma-separated list of weights (floats) corresponding to --metrics. "
              "If omitted equal weights are used."),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_name = args.model_name
    # metrics/weights support
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = ["accuracy", "auc", "recall"]

    weights = None
    if args.weights:
        try:
            weights = [float(w) for w in args.weights.split(",")]
        except Exception:
            logging.warning("Failed to parse weights; ignoring and using equal weights")
            weights = None

    metric = args.metric  # kept for backwards compatibility for single-metric remote path
    minimize = args.minimize
    local_only = args.local_only

    logging.info("Promoting best model for %s by metric %s (minimize=%s)", model_name, metric, minimize)

    reg = load_local_registry()

    # Try remote selection first unless local_only. Remote supports multi-metric now.
    chosen_version = None
    chosen_info = None
    if not local_only:
        sel = choose_best_remote(model_name, metrics, weights, minimize)
        if sel:
            chosen_version, chosen_info = sel
            logging.info("Selected remotely: version %s (info=%s)", chosen_version, chosen_info)

    # Fallback to local registry
    if chosen_version is None:
        sel = choose_best_from_local(reg, model_name, metrics, weights, minimize)
        if sel:
            chosen_version, chosen_info = sel
            logging.info("Selected from local registry: version %s (info=%s)", chosen_version, chosen_info)

    if not chosen_version:
        logging.error("No candidate model found for %s using metric %s", model_name, metric)
        return 2

    # Archive list - all versions except chosen
    all_versions = get_all_versions_local(reg, model_name)
    archive_versions = [v for v in all_versions if v != str(chosen_version)]

    # Try remote promotion (best-effort)
    remote_ok = False
    if not local_only:
        try:
            remote_ok = try_remote_promote(model_name, chosen_version, archive_versions)
            if remote_ok:
                logging.info("Remote registry updated")
        except Exception as e:
            logging.warning("Remote promotion attempt failed: %s", e)
            remote_ok = False

    # Always update local registry file to reflect desired state
    reg = update_local_registry_after_promotion(reg, model_name, chosen_version)
    save_local_registry(reg)

    # Friendly output
    logging.info("Promotion complete: model %s v%s set to Production locally", model_name, chosen_version)
    if not remote_ok:
        logging.info("Remote registry was not updated (either unavailable or operation failed). Local registry updated instead.")

    return 0


if __name__ == "__main__":
    print("final change")
    raise SystemExit(main())
