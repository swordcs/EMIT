import numpy as np
import pandas as pd
from autotune.utils.history_container import HistoryContainer
from autotune.utils.config_space.util import convert_configurations_to_array

MAX_MIN_SCALER = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
NUMERIC_METRICS = [  # counter
    # global
    "buffers_alloc",
    "buffers_backend",
    "buffers_backend_fsync",
    "buffers_checkpoint",
    "buffers_clean",
    "checkpoints_req",
    "checkpoints_timed",
    "checkpoint_sync_time",
    "checkpoint_write_time",
    "maxwritten_clean",
    "archived_count",
    "failed_count",
    # db
    "blk_read_time",
    "blks_hit",
    "blks_read",
    "blk_write_time",
    "conflicts",
    "deadlocks",
    "temp_bytes",
    "temp_files",
    "tup_deleted",
    "tup_fetched",
    "tup_inserted",
    "tup_returned",
    "tup_updated",
    "xact_commit",
    "xact_rollback",
    "confl_tablespace",
    "confl_lock",
    "confl_snapshot",
    "confl_bufferpin",
    "confl_deadlock",
    # table
    "analyze_count",
    "autoanalyze_count",
    "autovacuum_count",
    "heap_blks_hit",
    "heap_blks_read",
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "n_dead_tup",
    "n_live_tup",
    "n_tup_del",
    "n_tup_hot_upd",
    "n_tup_ins",
    "n_tup_upd",
    "n_mod_since_analyze",
    "seq_scan",
    "seq_tup_read",
    "tidx_blks_hit",
    "tidx_blks_read",
    "toast_blks_hit",
    "toast_blks_read",
    "vacuum_count",
    # index
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "idx_tup_read",
]


def extract_data(history_container):
    X = convert_configurations_to_array(history_container.configurations)
    y = history_container.get_transformed_perfs()
    resource = history_container.resource
    internal_metrics = history_container.internal_metrics

    # drop invalid data
    drop_indexes = set()
    for i in range(len(X)):
        if len(internal_metrics[i]) == 0:
            drop_indexes.add(i)
    X = pd.DataFrame([item for idx, item in enumerate(X) if idx not in drop_indexes])
    y = pd.DataFrame([item for idx, item in enumerate(y) if idx not in drop_indexes])
    y = y.apply(MAX_MIN_SCALER)
    resource_metric = resource[0].keys()
    resource = pd.DataFrame(
        [
            [item[k] for k in resource_metric]
            for idx, item in enumerate(resource)
            if idx not in drop_indexes
        ]
    )
    internal_metrics = pd.DataFrame(
        [item for idx, item in enumerate(internal_metrics) if idx not in drop_indexes]
    )

    X_col = [e.name for e in history_container.config_space.get_hyperparameters()]
    y_col = history_container.info["objs"]
    internal_metrics_col = NUMERIC_METRICS
    resource_col = [k for k in resource_metric]

    assert (
        len(X_col) == len(X.columns)
        and len(y_col) == len(y.columns)
        and len(resource_col) == len(resource.columns)
        and len(internal_metrics_col) == len(internal_metrics.columns)
    )

    X.columns = X_col
    y.columns = y_col
    internal_metrics.columns = internal_metrics_col
    resource.columns = resource_col

    return X, y, internal_metrics, resource
