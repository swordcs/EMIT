import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autotune.utils.constants import MAXINT

methodL = ["SMAC", "MBO", "DDPG", "GA"]
workloadL = ["twitter", "sysbench", "job", "tpch"]
spaceL = [197, 100, 50, 25, 12, 6]


def get_objs(res, y_variables):
    try:
        objs = []
        for y_variable in y_variables:
            key = y_variable.strip().strip("-")
            value = res[key]
            if not y_variable.strip()[0] == "-":
                value = -value
            objs.append(value)
    except:
        objs = [MAXINT]

    return objs[0]


def parse_data_onefile(fn):
    try:
        with open(fn) as fp:
            all_data = json.load(fp)
    except Exception as e:
        print(
            "Encountered exception %s while reading runhistory from %s. "
            "Not adding any runs!",
            e,
            fn,
        )
        return

    info = all_data["info"]
    data = all_data["data"]
    y_variables = info["objs"]
    objs, bests, configs = list(), list(), list()
    for tmp in data:
        em = tmp["external_metrics"]
        resource = tmp["resource"]
        res = dict(em, **resource)
        obj = get_objs(res, y_variables)
        objs.append(obj)
        config = tmp["configuration"]
        configs.append(config)
        if not len(bests) or obj < bests[-1]:
            bests.append(obj)
        else:
            bests.append(bests[-1])
    objs_remove_maxint = [item for item in objs if item < MAXINT]
    objs_lagest = max(objs_remove_maxint)
    objs_scaled = list()
    for obj in objs:
        if obj < MAXINT:
            objs_scaled.append(obj)
        else:
            objs.append(objs_lagest)

    return objs_scaled, bests, configs


def parse_data(file_dict):
    comparison = {}
    for method in file_dict.keys():
        comparison[method] = list()
        for file in file_dict[method]:
            objs, bests, _ = parse_data_onefile(file)
            comparison[method].append(
                {"objs": objs, "bests": bests, "n_calls": len(objs)}
            )
    return comparison


def get_best(file, iter):
    objs, bests, _ = parse_data_onefile(file)
    if len(bests) < iter:
        print("{} only has {} record, but require {}".format(file, len(bests), iter))
        iter = len(bests)

    return bests[iter - 1]


def plot_comparison(file_dict, workload, figname="plot/tmp.png", **kwargs):
    comparison = parse_data(file_dict)
    ax = plt.gca()
    ax.set_title("{} Convergence plot".format(workload))
    ax.set_xlabel(r"Number of iterations $n$")
    ax.grid()

    for i, method in enumerate(comparison.keys()):
        plot_scatter = False
        all_data = comparison[method]
        n_calls = np.max([_["n_calls"] for _ in all_data])
        bests = []
        if len(all_data) == 1:
            plot_scatter = True

        for data in all_data:
            if len(data["bests"]) < n_calls:
                print(
                    "{}th file for {} lacks {} record".format(
                        all_data.index(data), method, n_calls - len(data["best"])
                    )
                )
            while len(data["bests"]) < n_calls:
                data["bests"].append(data["bests"][-1])
            bests.append(data["bests"])

        iterations = range(1, int(n_calls) + 1)
        df_best_y = pd.DataFrame(bests)
        if bests[0][0] < 0:
            ax.set_ylabel(r"Throughput (txn/sec)")
            df_best_y = -df_best_y
            data["objs"] = [-item for item in data["objs"]]
        else:
            ax.set_ylabel(r"95th latency (sec)")

        min = df_best_y.quantile(0.25)
        max = df_best_y.quantile(0.75)
        mean = df_best_y.mean()

        # ax.plot(iterations, mean, color=tab10[i], label=method,  **kwargs)
        ax.plot(iterations, mean, color="C{}".format(i), label=method, **kwargs)
        ax.fill_between(iterations, min, max, alpha=0.2, color="C{}".format(i))
        # if plot_scatter:
        #    ax.scatter(iterations, data['objs'], color="C{}".format(i), )

    # plt.ylim(130, 160)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def plot_method(file_dict, workload):
    for method in methodL:
        file_dict_ = copy.deepcopy(file_dict)
        for key in file_dict.keys():
            if not method.upper() in key:
                del file_dict_[key]

        plot_comparison(
            file_dict_, workload, figname="plot/{}_{}.png".format(workload, method)
        )


def get_rank(file_dict, keyword, iter=200):
    file_dict_ = copy.deepcopy(file_dict)
    for key in file_dict.keys():
        if not str(keyword) in key and not str(keyword) in file_dict_[key][0]:
            del file_dict_[key]

    performance = np.zeros(4)
    for key in file_dict_.keys():
        per = get_best(file_dict_[key][0], iter)
        method = key.split("-")[1]
        performance[methodL.index(method)] = per

    return performance.argsort().argsort()


def plot_rank(
    file_dict, type="one-workload", keyword="", iter=200, figname="plot/tmp.png"
):
    rankL = list()
    if not keyword == "":
        file_dict_ = copy.deepcopy(file_dict)
        for key in file_dict.keys():
            if not keyword in key:
                del file_dict_[key]

    if type == "one-workload":
        filterL = spaceL
    elif type == "one-space":
        filterL = workloadL
    else:
        filterL1 = workloadL
        filterL2 = spaceL

    if type in ["one-workload", "one-space"]:
        for item in filterL:
            rank = get_rank(file_dict_, item, iter)
            rankL.append(rank)
            print(rank)
    else:
        for item1 in filterL1:
            file_dict_ = copy.deepcopy(file_dict)
            for key in file_dict.keys():
                if not item1 in key:
                    del file_dict_[key]
            for item2 in filterL2:
                rank = get_rank(file_dict_, item2, iter)
                rankL.append(rank)
                print(rank)

    rankL_T = np.array(rankL).T.tolist()
    ax = plt.gca()
    ax.grid()
    ax.boxplot(rankL_T)
    plt.title(keyword)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(methodL)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


if __name__ == "__main__":
    pass
