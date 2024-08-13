# use package without intalling
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
import time

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.ini", help="config file")
    opt = parser.parse_args()

    args_db, args_tune, args_common = parse_args(opt.config)

    if args_db["db"] == "mysql":
        db = MysqlDB(args_db)
    elif args_db["db"] == "postgresql":
        db = PostgresqlDB(args_db)
    if args_common.get("multi_tune", False) == "True":
        for i in range(int(args_common.get("max_runs", 0))):
            args_db, args_tune, _ = ch_config_for_collection(
                args_db=args_db,
                args_tune=args_tune,
                args_common=args_common,
                target=("weights", "task_id", "db_log"),
            )
            env = DBEnv(args_db, args_tune, db)
            tuner = DBTuner(args_db, args_tune, env)
            tuner.tune()
    elif (
        args_common.get("workload_synthesis", False) == "True"
        and args_common.get("multi_exp", False) == "True"
    ):
        task = None
        with open("share/workload_synthesis_for_exp.json", "r") as f:
            workload_synthesis = json.load(f)

        for workload, configs in task.items():
            for config in configs:
                args_tune["transfer_framework"] = config["transfer_framework"]
                args_tune["acq_type"] = config["acq_type"]
                args_tune["initial_runs"] = config["initial_runs"]
                args_db["dbname"] = workload.split("_")[0]
                db = PostgresqlDB(args_db)
                weight = workload_synthesis[workload]["weight"]
                source_id = workload_synthesis[workload]["source_id"]
                args_tune["tasks"] = [
                    source_id[i].replace("history_task_SMAC_", "") + "_%.2f" % weight[i]
                    for i in range(len(weight))
                ]
                for i in range(config["repetition"]):
                    args_db["oltpbench_config_xml"] = (
                        f"/home/gengj/Project/DBTune/config/{workload}_config.xml"
                    )
                    args_tune["task_id"] = (
                        f"synthesis_{args_tune['optimize_method']}_{workload}_{config['acq_type']}_{int(time.time())}"
                    )
                    env = DBEnv(args_db, args_tune, db)
                    tuner = DBTuner(args_db, args_tune, env)
                    tuner.tune()
                time.sleep(300)

    elif args_common.get("workload_synthesis", False) == "True":
        with open("share/workload_synthesis.json", "r") as f:
            workload_synthesis = json.load(f)
        for key, value in workload_synthesis.items():
            target_task = key
            source_task = value["source_id"]
            wkld_weight = value["weight"]
            ch_config_for_synthsis(
                target=key,
                source=value,
                args_tune=args_tune,
            )
            env = DBEnv(args_db, args_tune, db)
            tuner = DBTuner(args_db, args_tune, env)
            tuner.tune()
    elif args_common.get("multi_exp", False) == "True":
        common_task = {
            "acq_type": "ei",
            "initial_runs": 10,
            "repetition": 1,
        }
        task1 = None
        task2 = None
        task3 = None
        task4 = None
        task_set = {"task1": task1, "task2": task2, "task3": task3, "task4": task4}
        task = task_set[args_common["task_name"]]
        print(f"starting {args_common['task_name']}")
        for key, value in task.items():
            for v in value:
                args_tune["transfer_framework"] = v["transfer_framework"]
                args_tune["acq_type"] = v["acq_type"]
                args_tune["initial_runs"] = v["initial_runs"]
                args_tune["optimize_method"] = v["optimize_method"]
                args_tune["initial_tunable_knob_num"] = v.get(
                    "initial_tunable_knob_num", 30
                )
                args_db["dbname"] = key.split("_")[0]
                db = PostgresqlDB(args_db)

                task_id_done = set()
                for i in range(v["repetition"]):
                    args_db["oltpbench_config_xml"] = (
                        f"/home/gengj/Project/DBTune/config/{key}_config.xml"
                    )
                    if "task_id" in v.keys() and v["task_id"] not in task_id_done:
                        args_tune["task_id"] = v["task_id"]

                    else:
                        args_tune["task_id"] = (
                            f"{args_tune['optimize_method']}_{args_tune['transfer_framework']}_{args_tune['initial_tunable_knob_num']}knob_{key}_{v['acq_type']}_{int(time.time())}"
                        )
                    task_id_done.add(args_tune["task_id"])
                    ch_config_for_experiement(
                        args_db=args_db, args_tune=args_tune, args_common=args_common
                    )
                    env = DBEnv(args_db, args_tune, db)
                    tuner = DBTuner(args_db, args_tune, env)
                    tuner.tune()
                time.sleep(300)
    elif args_common.get("experiment", False) == "True":

        ch_config_for_experiement(
            args_db=args_db, args_tune=args_tune, args_common=args_common
        )
        env = DBEnv(args_db, args_tune, db)
        tuner = DBTuner(args_db, args_tune, env)
        tuner.tune()
    else:
        env = DBEnv(args_db, args_tune, db)
        tuner = DBTuner(args_db, args_tune, env)
        tuner.tune()
