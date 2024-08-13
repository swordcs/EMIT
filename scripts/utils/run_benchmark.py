import os
import sys
import time
import glob
import json
import numpy as np
import argparse
import subprocess
import multiprocessing as mp
import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from skopt import forest_minimize

sys.path.append(".")

from string import Template
from multiprocessing import Manager
from scripts.utils.docker_monitor import DockerMonitor
from autotune.database.postgresqldb import PostgresqlDB
from autotune.utils.config import parse_args
from autotune.utils.parser import parse_oltpbench


BENCHMARK_RUNNING_TIME = 0
BENCHMARK_WARMING_TIME = 0


def _benchmark_cmd(config_xml, dbname):
    filename = os.path.join(os.getcwd(), f"results/{int(time.time_ns())}")
    bash_script = "/home/gengj/Project/DBTune/autotune/cli/run_oltpbench.sh"

    cmd = "bash {} {} {} {} {} {} {}".format(
        bash_script,
        "benchbase-postgres",
        dbname,
        config_xml,
        filename,
        "--execute=true",
        "",
    )

    return cmd, filename


def generate_benchmark_cmd(
    benchmark, weight, args_db, rate="unlimited", warmup=30, time=120
):

    global BENCHMARK_RUNNING_TIME
    global BENCHMARK_WARMING_TIME

    BENCHMARK_RUNNING_TIME = time
    BENCHMARK_WARMING_TIME = warmup

    config_dir = "/home/gengj/Project/DBTune/config"
    template_file = f"{config_dir}/template_{benchmark}_config.xml"
    args_db["dbname"] = benchmark
    weight = [str(_) for _ in weight]
    with open(
        template_file,
        "r",
    ) as f:
        src = Template(f.read())
        context = src.substitute(
            {
                "weights": ",".join(weight),
                "warmup": str(warmup),
                "time": str(time),
                "dbport": str(args_db["port"]),
                "rate": str(rate),
            }
        )

    template_file = f"{config_dir}/tmp/{benchmark}_{'_'.join(weight)}.xml"
    with open(template_file, "w") as f:
        f.write(context)
    cmd, filename = _benchmark_cmd(config_xml=template_file, dbname=benchmark)
    return cmd, filename


def parse_result(filename):
    for _ in range(60):
        if os.path.exists(filename):
            break
        time.sleep(1)
    file_list = sorted(list(glob.glob(os.path.join(filename, "*summary*"))))
    if len(file_list) == 0:
        print("benchmark result file does not exist!")
    return parse_oltpbench(file_list[-1])


def run_benchmark(cmd, filename, database, collect_resource=True):
    print("benchmark start!")
    benchmark_timeout = False

    if isinstance(cmd, str):
        cmd = [cmd]

    p_benchmarks = [
        subprocess.Popen(
            c,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            close_fds=True,
        )
        for c in cmd
    ]

    # Internal Metrics & Resource Consumption Collection ↓
    # start Internal Metrics Collection
    internal_metrics = Manager().list()
    im = mp.Process(
        target=database.get_internal_metrics,
        args=(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME),
    )
    database.set_im_alive(True)
    im.start()

    if collect_resource:
        rm = DockerMonitor(
            1, BENCHMARK_WARMING_TIME, BENCHMARK_RUNNING_TIME, str(database.port)[0]
        )
        rm.run()

    try:
        [p.communicate(timeout=600) for p in p_benchmarks]

        ret_codes = [p.poll() for p in p_benchmarks]
        if all([_ == 0 for _ in ret_codes]):
            print("benchmark finished!")
        else:
            print("run benchmark get error {}".format(ret_codes))
            return

    except subprocess.TimeoutExpired:
        benchmark_timeout = True
        print("benchmark timeout!")

    # stop Internal Metrics Collection
    database.set_im_alive(False)
    im.join()

    if collect_resource:
        rm.terminate()
        cpu, mem, read, write = rm.get_monitor_data()

    if isinstance(filename, str):
        filename = [filename]

    external_metrics = [parse_result(fn) for fn in filename]
    internal_metrics, _, _, _ = database._post_handle(internal_metrics)

    res = {
        "cmds": cmd,
        "external_metrics": list(external_metrics),
        "internal_metrics": list(internal_metrics),
        "resource": {
            "cpu": list(cpu),
            "mem": list(mem),
            "read": list(read),
            "write": list(write),
        },
    }

    with open("benchmark_metrics.json", "a") as f:
        f.write(json.dumps(res) + "\n")

    return (
        benchmark_timeout,
        list(np.sum(external_metrics, axis=0)),
        list(internal_metrics),
        (
            cpu,
            mem,
            read,
            write,
        ),
    )


def load_basic_workload(path, target=None):
    data = {}
    target_data = None
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            tmp = json.loads(line)
            if target is None:
                data.update(tmp)
                continue

            if tmp.get(target):
                target_data = tmp[target]
            elif (
                "base" in list(tmp.keys())[0]
                and target.split("_")[0] not in list(tmp.keys())[0]
            ):
                data.update(tmp)
    return data, target_data


def workload_similarity(v1, v2, method="o"):
    v1, v2 = np.array(v1), np.array(v2)
    if method.upper() == "O":
        return np.sqrt(np.sum((v1 - v2) ** 2))
    if method.upper() == "COSINE":
        return np.sum(v1 * v2) / (np.sqrt(np.sum(v1**2) * np.sum(v2**2)))


def get_PCAer(data, n_comp):
    pca_opt = PCA(n_components=n_comp)
    pca_opt.fit(data)
    return pca_opt


def get_STDer(data):
    std_opt = StandardScaler(with_mean=False)
    std_opt.fit(data)
    return std_opt


def get_processor(pca_comp=None):
    base_data, _ = load_basic_workload("benchmark_result.json")
    base_im = [x["internal_metrics"] for x in base_data.values()]
    base_rs = [[np.mean(_) for _ in x["resource"].values()] for x in base_data.values()]
    im_std = get_STDer(base_im)
    if pca_comp is None:
        pca_comp = len(base_rs[0])
    im_pca = get_PCAer(im_std.transform(base_im), pca_comp)
    rs_std = get_STDer(base_rs)
    rs_pca = get_PCAer(rs_std.transform(base_rs), pca_comp)
    return im_std, im_pca, rs_std, rs_pca, base_im, base_rs


def workload_synthesis(target, im_std, im_pca, rs_std, num=2):

    base_data, target_data = load_basic_workload("benchmark_result.json", target)

    base_im = [x["internal_metrics"] for x in base_data.values()]
    base_rs = [[np.mean(_) for _ in x["resource"].values()] for x in base_data.values()]
    source_id = list(base_data.keys())

    # internal metrics standardize and PCA process
    base_im = im_pca.transform(im_std.transform(base_im))
    target_im = im_pca.transform(im_std.transform([target_data["internal_metrics"]]))

    # resource standardize process
    base_rs = rs_std.transform(base_rs)
    target_rs = rs_std.transform(
        [[np.mean(_) for _ in target_data["resource"].values()]]
    )

    base_metric = np.hstack((base_im, base_rs))
    target_metric = np.hstack((target_im, target_rs))

    A = np.array(base_metric).T
    d = np.array(target_metric)

    def penalty(x, nonzero=1):
        return 40 * (np.count_nonzero(x > 1e-1) - nonzero) ** 2

    def objective(nonzero=1, method="cosine"):
        if method == "cosine":
            return lambda x: -np.sum((A @ x) * d) / (
                np.sqrt(np.sum((A @ x) ** 2) * np.sum(d**2))
            ) + penalty(x, nonzero)
        if method == "o":
            return lambda x: np.sqrt(np.sum((A @ x - d) ** 2)) + penalty(x, nonzero)

    base_similarity = np.array(
        [workload_similarity(x, target_im, method="cosine") for x in base_im]
    )

    b = np.zeros_like(base_similarity)
    b[base_similarity == base_similarity.max()] = 1
    con = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1,
    }

    x = minimize(
        objective(nonzero=num, method="o"),
        b,
        bounds=[(0, 10)] * A.shape[1],
        # constraints=con,
    ).x

    x[x < 1e-1] = 0
    x = x / x.sum()

    cos_similarity = workload_similarity(A @ x, d, method="cosine")
    eu_similarity = workload_similarity(A @ x, d, method="o")

    print(target)
    print(
        f"cosine similarity: {cos_similarity},  euclidean similarity: {eu_similarity}"
    )
    print(x[x > 0].tolist(), np.array(source_id)[x > 0].tolist())
    print("=====================================")

    return x[x > 0].tolist(), np.array(source_id)[x > 0].tolist(), A @ x, d


def test_collect_data():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file")
    opt = parser.parse_args()

    args_db, args_tune, args_common = parse_args(opt.config)

    # tasks = {
    #     "tpcc": [
    #         (100, 0, 0, 0, 0),
    #         (0, 100, 0, 0, 0),
    #         (0, 0, 100, 0, 0),
    #         (0, 0, 0, 100, 0),
    #         (0, 0, 0, 0, 100),
    #     ],
    #     "ycsb": [
    #         (100, 0, 0, 0, 0, 0),
    #         (0, 100, 0, 0, 0, 0),
    #         (0, 0, 100, 0, 0, 0),
    #         (0, 0, 0, 100, 0, 0),
    #         (0, 0, 0, 0, 100, 0),
    #     ],
    #     "twitter": [
    #         (100, 0, 0, 0, 0),
    #         (0, 100, 0, 0, 0),
    #         (0, 0, 100, 0, 0),
    #         (0, 0, 0, 100, 0),
    #         (0, 0, 0, 0, 100),
    #     ],
    # }

    tasks = {
        "tpcc": [(45, 43, 4, 4, 4)],
        "ycsb": [(50, 0, 0, 50, 0, 0), (95, 0, 0, 95, 0, 0)],
        "twitter": [(1, 1, 7, 1, 90)],
    }

    for benchmark, weights in tasks.items():
        for weight in weights:
            cmd, filename = generate_benchmark_cmd(
                benchmark, weight, args_db, warmup=30, time=300
            )
            db = PostgresqlDB(args_db)
            db.restart()
            timeout, external_metrics, internal_metrics, resource = run_benchmark(
                cmd, filename, db
            )
            res = {
                "benchmark": benchmark,
                "weight": weight,
                "external_metrics": external_metrics,
                "internal_metrics": internal_metrics,
                "resource": {
                    "cpu": resource[0],
                    "mem": resource[1],
                    "read": resource[2],
                    "write": resource[3],
                },
            }

            with open("benchmark_result.json", "a") as f:
                f.write(
                    json.dumps(
                        {f"{benchmark}_{'_'.join([str(_) for _ in weight])}": res}
                    )
                    + "\n"
                )
            print(f"{benchmark}_{'_'.join([str(_) for _ in weight])} is saved.")
            print("wait for cooling down...")
            time.sleep(120)


def test_workload_synthesis():
    im_std, im_pca, rs_std, rs_pca, base_im, base_rs = get_processor()
    res = {}
    for target in ["tpcc", "ycsb_a", "ycsb_b", "twitter"]:
        weights, ids, syned_data, target_data = workload_synthesis(
            target, im_std, im_pca, rs_std, num=2
        )
        res[target] = {
            "weights": weights,
            "ids": ids,
            "syned_data": syned_data,
            "target_data": target_data,
        }

    base_im = im_pca.transform(im_std.transform(base_im))
    base_rs = rs_pca.transform(rs_pca.transform(base_rs))

    base_metric = np.hstack((base_im, base_rs))

    pca = PCA(n_components=2)
    pca.fit(base_metric)

    base_metric = pca.transform(base_metric[:-4])
    targ_metric = pca.transform([x["target_data"][0] for x in res.values()])
    synd_metric = pca.transform([x["syned_data"] for x in res.values()])

    import matplotlib.pyplot as plt

    base_x, base_y = base_metric[:, 0], base_metric[:, 1]
    targ_x, targ_y = targ_metric[:, 0], targ_metric[:, 1]
    synd_x, synd_y = synd_metric[:, 0], synd_metric[:, 1]
    plt.scatter(base_x, base_y, label="base", s=5)
    plt.scatter(targ_x, targ_y, label="target", s=5, c=["r", "green", "b", "y"])
    plt.scatter(synd_x, synd_y, label="synthesized", s=5, c=["r", "green", "b", "y"])
    plt.legend()
    plt.savefig("synthesized.png")


TASKS = []
PROCESSOR = {
    "im_std": None,
    "im_pca": None,
    "rs_std": None,
}

TARGET_METRIC = None


def workload_synthesis_obj(rates):

    time.sleep(10)

    global TASKS, PROCESSOR
    cmds = []
    files = []
    for task in TASKS:
        cmd, filename = generate_benchmark_cmd(
            task["workload"],
            task["weight"],
            task["args_db"],
            rates[TASKS.index(task)] * 100,
            warmup=300,
            time=300,
        )
        cmds.append(cmd)
        files.append(filename)

    PostgresqlDB(task["args_db"]).restart()
    obj = run_benchmark(cmds, files, PostgresqlDB(task["args_db"]))
    if obj is None:
        return 100

    _, _, internal_metrics, resource = obj

    im_std = PROCESSOR["im_std"]
    im_pca = PROCESSOR["im_pca"]
    rs_std = PROCESSOR["rs_std"]

    im = im_pca.transform(im_std.transform([internal_metrics]))
    rs = rs_std.transform([[np.mean(_) for _ in resource]])
    metric = np.hstack((im, rs))[0]
    return workload_similarity(TARGET_METRIC, metric, method="o")


def test_BO_synthesis(target, args_db):
    with open("synthesis.json", "r") as f:
        syn_res = json.load(f)

    tmp = syn_res[target]
    workloads = [s[: s.index("+")] for s in tmp["workload"]]
    weights = [s[s.index("+") + 1 :].split("_") for s in tmp["workload"]]
    im_std, im_pca, rs_std, _, _, _ = get_processor()

    _, target_data = load_basic_workload("benchmark_result.json", target)

    target_im = im_pca.transform(im_std.transform([target_data["internal_metrics"]]))
    # resource standardize process
    target_rs = rs_std.transform(
        [[np.mean(_) for _ in target_data["resource"].values()]]
    )

    global TARGET_METRIC
    TARGET_METRIC = np.hstack((target_im, target_rs))[0]

    task = []
    for workload, weight in zip(workloads, weights):
        task.append(
            {
                "workload": workload,
                "weight": weight,
                "args_db": args_db,
            }
        )

    global TASKS
    TASKS = task

    global PROCESSOR

    PROCESSOR["im_std"] = im_std
    PROCESSOR["im_pca"] = im_pca
    PROCESSOR["rs_std"] = rs_std

    res = forest_minimize(workload_synthesis_obj, [(1, 40), (1, 40)], n_calls=50)
    print(f"target:{target}, workloads:{workloads}, weights:{res.x}")
    return res


def verify_synthesis():
    with open("benchmark_metrics.json", "r") as f:
        metrics = json.load(f)

    im_std, im_pca, rs_std, _, _, _ = get_processor()

    im = [x["internal_metrics"] for x in metrics]
    rs = [[np.mean(_) for _ in x["resource"].values()] for x in metrics]

    im = im_pca.transform(im_std.transform(im))
    rs = rs_std.transform(rs)

    metric = np.hstack((im, rs))

    syn_res, _ = load_basic_workload("benchmark_result.json")

    similar_res = dict()

    for target in ["tpcc", "ycsb_a", "ycsb_b", "twitter"]:
        syns = syn_res[target]
        im_target = im_pca.transform(im_std.transform([syns["internal_metrics"]]))
        rs_target = rs_std.transform([[np.mean(_) for _ in syns["resource"].values()]])

        metric_target = np.hstack((im_target, rs_target))[0]

        tmp_similarity = []
        for i in range(len(metric)):
            tmp_similarity.append(
                {
                    "eu": workload_similarity(metric[i], metric_target, method="o"),
                    "cosine": workload_similarity(
                        metric[i], metric_target, method="cosine"
                    ),
                    "rs":{
                        "eu": workload_similarity(metric[i][-4:], metric_target[-4:], method="o"),
                        "cosine": workload_similarity(metric[i][-4:], metric_target[-4:], method="o")
                    },
                    "im":{
                        "eu": workload_similarity(metric[i][:4], metric_target[:4], method="o"),
                        "cosine": workload_similarity(metric[i][:4], metric_target[:4], method="o")
                    }
                }
            )
        similar_res[target] = tmp_similarity
    return similar_res

