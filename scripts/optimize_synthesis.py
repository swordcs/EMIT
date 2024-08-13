# use package without intalling
import math
import os
import sys
import argparse
import json
import numpy as np
import paramiko


sys.path.append(".")

from string import Template
from autotune.dbenv import DBEnv
from autotune.knobs import logger
from autotune.tuner import DBTuner
from autotune.utils.config import parse_args
from autotune.utils.parser import ConfigParser
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.utils.history_container import HistoryContainer
from autotune.utils.config_space.util import convert_configurations_to_array

TUNED_WEIGHTS = set()
TEMPLATE_PATH = ""


def ch_config_for_collection(
    args_db=None, args_tune=None, args_common=None, target=None, weight=None
):
    # sample transaction weights for benchmark
    global TUNED_WEIGHTS
    global TEMPLATE_PATH
    if "weights" in target:
        if weight is None:
            weights_num = 6
            total_weights = 100
            weights = tuple(
                np.random.multinomial(
                    total_weights / 20, np.ones(weights_num) / weights_num
                )
                * 20
            )
            while weights in TUNED_WEIGHTS:
                weights = tuple(
                    np.random.multinomial(
                        total_weights / 20, np.ones(weights_num) / weights_num
                    )
                    * 20
                )
        else:
            weights = weight
        TUNED_WEIGHTS.add(weights)

    # generate task id
    if "task_id" in target:
        task_id = (
            "task_"
            + args_tune["optimize_method"]
            + "_"
            + args_db["dbname"]
            + "_"
            + "_".join([str(_) for _ in weights])
        )
        args_tune["task_id"] = task_id

    if "weights" in target:
        # produce xml config for benchbase
        if TEMPLATE_PATH == "":
            TEMPLATE_PATH = args_db["oltpbench_config_xml"]
        with open(
            TEMPLATE_PATH,
            "r",
        ) as f:
            src = Template(f.read())
            context = src.substitute(
                {
                    "weights": ",".join([str(_) for _ in weights]),
                    "warmup": args_db["workload_warmup_time"],
                    "time": args_db["workload_time"],
                }
            )
        tmp_xml_template = os.path.dirname(TEMPLATE_PATH) + f"/tmp/{task_id}.xml"
        with open(tmp_xml_template, "w") as f:
            f.write(context)
        args_db["oltpbench_config_xml"] = tmp_xml_template

    # change db log path
    if "db_log" in target:
        cnf = args_db["cnf"]
        if args_db.get("remote_mode", False):
            cnf = "/tmp/pglocal.cnf"
            ssh_pk_file = os.path.expanduser(args_db["private_key"])
            pk = paramiko.Ed25519Key.from_private_key_file(ssh_pk_file)
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                args_db.get("host"),
                username=args_db.get("ssh_user"),
                pkey=pk,
                port=args_db.get("ssh_port"),
                disabled_algorithms={"pubkeys": ["rsa-sha2-256", "rsa-sha2-512"]},
            )
            sftp = ssh.open_sftp()
            try:
                sftp.get(args_db["cnf"], cnf)
            except IOError:
                logger.info("download PGCNF failed!")

        cnf_parser = ConfigParser(cnf)
        cnf_parser.set("log_filename", f"'postgresql_{task_id}.log'")
        cnf_parser.replace()

        if args_db.get("remote_mode", False):
            try:
                sftp.put(cnf, args_db["cnf"])
            except IOError:
                logger.info("upload PGCNF failed!")
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()

    return args_db, args_tune, args_common


def ch_config_for_synthsis(target, source, args_tune=None):
    task_id = (
        "synthesis_"
        + args_tune["optimize_method"]
        + "_"
        + target.replace("history_task_SMAC_", "")
    )
    source_ids = source["source_id"]
    weight = source["weight"]

    args_tune["task_id"] = task_id
    args_tune["tasks"] = [
        source_ids[i].replace("history_task_SMAC_", "") + "_%.2f" % weight[i]
        for i in range(len(weight))
    ]


def ch_config_for_experiement(
    args_db=None, args_tune=None, args_common=None, log_config=False
):
    # db_log configuration
    task_id = args_tune["task_id"]
    global TEMPLATE_PATH
    # if TEMPLATE_PATH == "":
    TEMPLATE_PATH = args_db["oltpbench_config_xml"]
    with open(
        TEMPLATE_PATH,
        "r",
    ) as f:
        src = Template(f.read())
        context = src.substitute(
            {
                "warmup": args_db["workload_warmup_time"],
                "time": args_db["workload_time"],
                "dbport": args_db["port"],
            }
        )
    tmp_xml_template = os.path.dirname(TEMPLATE_PATH) + f"/tmp/{task_id}.xml"
    with open(tmp_xml_template, "w") as f:
        f.write(context)
    args_db["oltpbench_config_xml"] = tmp_xml_template
    if not log_config:
        return

    cnf = args_db["cnf"]
    if args_db.get("remote_mode", False):
        cnf = "/tmp/pglocal.cnf"
        ssh_pk_file = os.path.expanduser(args_db["private_key"])
        pk = paramiko.Ed25519Key.from_private_key_file(ssh_pk_file)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            args_db.get("host"),
            username=args_db.get("ssh_user"),
            pkey=pk,
            port=args_db.get("ssh_port"),
            disabled_algorithms={"pubkeys": ["rsa-sha2-256", "rsa-sha2-512"]},
        )
        sftp = ssh.open_sftp()
        try:
            sftp.get(args_db["cnf"], cnf)
        except IOError:
            logger.info("download PGCNF failed!")

    cnf_parser = ConfigParser(cnf)

    cnf_parser.set("log_filename", f"'postgresql_{task_id}.log'")
    cnf_parser.replace()

    if args_db.get("remote_mode", False):
        try:
            sftp.put(cnf, args_db["cnf"])
        except IOError:
            logger.info("upload PGCNF failed!")
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()


KNOBS_INFO = "scripts/experiment/gen_knobs/postgres_new.json"
from sklearn.cluster import KMeans


def recursive_selection(X, y, indices, all_configs, n_cluster=3, env=None, log_fd=None, importance=None):
    logs = []
    i = 0
    while True:
        i = i + 1
        print(f"Iteration {i}")
        log_fd.write(f"Iteration {i}")
        log_fd.flush()
        if len(X) <= n_cluster:
            best_config, log = select_configuration_sequently(configs, env, log_fd)
            logs = logs + log
            return best_config, logs

        kmeans = KMeans(n_clusters=3).fit(X * importance)

        labels = []
        configs = []
        for j in range(len(X)):
            if kmeans.labels_[j] not in labels:
                labels.append(kmeans.labels_[j])
                configs.append(all_configs[indices[j]])
        best_config, log = select_configuration_sequently(configs, env, log_fd)
        if best_config is None:
            return None, None
        logs = logs + log
        best_cluster = labels[configs.index(best_config)]
        X = X[kmeans.labels_ == best_cluster]
        y = y[kmeans.labels_ == best_cluster]
        indices = indices[kmeans.labels_ == best_cluster]


def select_configuration_sequently(configs, env, log_fd=None):
    best_perf = -float("inf")
    log = []
    best_config = None
    for config in configs:
        # for _ in range(5):
        try:
            _, external_metrics, _, _ = env.step_GP(config)
        except:
            continue
        if external_metrics[0] > best_perf:
            log_fd.write(f"performance found: {external_metrics[0]}, config: {config}")
            log_fd.flush()

            print("performance found: ", external_metrics[0], "config: ", config)
            log.append(external_metrics[0])
            best_perf = external_metrics[0]
            best_config = config
        else:
            break
    return best_config, log


WORKLOAD_TYPE = {"tpcc", "ycsb_A", "ycsb_B", "twitter"}

import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.ini", help="config file")
    opt = parser.parse_args()

    args_db, args_tune, args_common = parse_args(opt.config)

    tasks = []
    for task in tasks:

        for wt in WORKLOAD_TYPE:
            if wt in task:
                break

        print(f"Testing : {wt}")

        args_db["dbname"] = wt.split("_")[0]
        args_db["oltpbench_config_xml"] = (
            f"/home/gengj/Project/DBTune/config/test/{wt}_config_test.xml"
        )
        args_tune["task_id"] = f"Synthesis_test_{wt}_{time.time()}"

        db = PostgresqlDB(args_db)

        config_space = DBTuner.setup_configuration_space(KNOBS_INFO, -1)
        history_container = HistoryContainer(0, config_space=config_space)
        history_container.load_history_from_json(task)

        env = DBEnv(args_db, args_tune, db)

        X = convert_configurations_to_array(history_container.configurations)
        y = np.array(history_container.get_all_perfs())

        from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter, Constant
        from sklearn.preprocessing import LabelEncoder


        importance = {}


        columns = history_container.config_space_all.get_hyperparameter_names()
        X = pd.DataFrame(history_container.configurations_all)
        X = X[columns]

        le = LabelEncoder()
        for col in list(X.columns):
            if isinstance(history_container.config_space_all[col], CategoricalHyperparameter):
                le.fit(X[col])
                X[col] = le.transform(X[col])
            else:
                X[col] = X[col].astype('float')

        Y = np.array(history_container.get_transformed_perfs()).astype('float')

        f = fANOVA(X=X, Y=Y, config_space=config_space)

        importance = []
        for i in list(X.columns):
            value = f.quantify_importance((i,))[(i,)]['individual importance']
            if not math.isnan(value):

                importance.append(value)

        sorted_indices = np.argsort(y)[:27]

        watershed = 27
        X = X[sorted_indices]
        y = y[sorted_indices]

        LOG_FD = open(f"synthesis_test_{wt}_{time.time()}.log", "a")

        best_config, logs = recursive_selection(
            X,
            y,
            sorted_indices,
            history_container.configurations,
            3,
            env,
            log_fd=LOG_FD,
            importance=importance
        )
