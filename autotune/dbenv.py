import os
import sys
import glob
import time
import subprocess
import numpy as np
import multiprocessing as mp
from .knobs import logger
from .knobs import initialize_knobs, get_default_knobs
from .utils.parser import parse_sysbench, parse_oltpbench, parse_job
from .resource_monitor import ResourceMonitor
from string import Template
from multiprocessing import Manager
from multiprocessing.connection import Client
from autotune.workload import (
    SYSBENCH_WORKLOAD,
    JOB_WORKLOAD,
    OLTPBENCH_WORKLOADS,
    TPCH_WORKLOAD,
)
from autotune.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from autotune.utils.parser import is_number
from autotune.database.postgresqldb import PostgresqlDB


class DBEnv:
    def __init__(self, args_db, args_tune, db):
        self.db = db
        self.args_db = args_db
        self.args_tune = args_tune
        self.workload = self.get_workload()
        self.log_path = "./log"
        self.num_metrics = self.db.num_metrics
        self.threads = int(args_db["thread_num"])
        self.knobs_detail = initialize_knobs(
            args_db["knob_config_file"], int(args_db["knob_num"])
        )
        self.default_knobs = get_default_knobs()
        self.online_mode = eval(args_db["online_mode"])
        self.remote_mode = eval(args_db["remote_mode"])
        self.oltpbench_config_xml = args_db["oltpbench_config_xml"] # change in multi run
        self.step_count = 0
        self.reinit_interval = 0
        self.reinit = False
        self.data_loaded = False
        self.connect_sucess = True
        if self.reinit_interval:
            self.reinit = False
        self.generate_time()
        self.y_variable = eval(args_tune["performance_metric"])

        self.reference_point = self.generate_reference_point(
            eval(args_tune["reference_point"])
        )

        if args_tune["constraints"] is None or args_tune["constraints"] == "":
            self.constraints = []
        else:
            self.constraints = eval(args_tune["constraints"])
        self.lhs_log = args_db["lhs_log"]
        self.cpu_core = args_db["cpu_core"]
        self.info = {"objs": self.y_variable, "constraints": self.constraints}

    def generate_reference_point(self, user_defined_reference_point):
        if len(self.y_variable) <= 1:
            return None

        reference_point_dir = {
            "tps": 0,
            "lat": BENCHMARK_RUNNING_TIME,
            "qps": 0,
            "cpu": 0,
            "readIO": 0,
            "writeIO": 0,
            "virtualMem": 0,
            "physical": 0,
        }
        reference_point = []
        for key in self.y_variable:
            use_defined_value = user_defined_reference_point[self.y_variable.index(key)]
            if is_number(use_defined_value):
                reference_point.append(use_defined_value)
            else:
                key = key.strip().strip("-")
                reference_point.append(reference_point_dir[key])

        return reference_point

    def get_workload(self):
        if self.args_db["workload"].startswith("oltpbench_"):
            wl = dict(OLTPBENCH_WORKLOADS)
        else:
            raise ValueError("Invalid workload!")
        return wl

    def generate_time(self):
        global BENCHMARK_RUNNING_TIME
        global BENCHMARK_WARMING_TIME
        global TIMEOUT_TIME
        global RESTART_FREQUENCY

        if self.workload["name"] == "oltpbench":
            try:
                BENCHMARK_RUNNING_TIME = int(self.args_db["workload_time"])
            except:
                BENCHMARK_RUNNING_TIME = 120
            try:
                BENCHMARK_WARMING_TIME = int(self.args_db["workload_warmup_time"])
            except:
                BENCHMARK_WARMING_TIME = 30
            TIMEOUT_TIME = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME + 30
            RESTART_FREQUENCY = 200
        else:
            raise ValueError("Invalid workload nmae!")

    def get_external_metrics(self, filename=""):
        if self.workload["name"] == "oltpbench":
            for _ in range(60):
                if os.path.exists(filename):
                    break
                time.sleep(1)
            file_list = sorted(list(glob.glob(os.path.join(filename, "*summary*"))))
            if len(file_list) == 0:
                print("benchmark result file does not exist!")
            result = parse_oltpbench(file_list[-1])
        else:
            raise ValueError("Invalid workload name!")
        return result

    def get_benchmark_cmd(self, oltpbench_config_xml=None):
        timestamp = int(time.time())
        filename = self.log_path + "/{}.log".format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        if self.workload["name"] == "oltpbench":
            filename = os.path.join(os.getcwd(), f"results/{timestamp}")
            if oltpbench_config_xml is None:
                oltpbench_config_xml = self.oltpbench_config_xml
            if isinstance(self.db, PostgresqlDB):
                benchbase = "benchbase-postgres"
            cmd = self.workload["cmd"].format(
                dirname + "/cli/run_oltpbench.sh",
                benchbase,
                self.db.dbname,
                oltpbench_config_xml,
                filename,
                "--execute=true",
                "",
            )
        else:
            raise ValueError("Invalid workload name!")

        return cmd, filename

    def load_benchmark_data(self, timeout=1800):
        cmd = ""
        if self.workload["name"] == "oltpbench":
            cmd = self.get_benchmark_cmd()[0]
            cmd = cmd.replace("--execute=true", "--create=true --load=true")
        # start load benchmark data
        logger.info("load data start!")

        p_loaddata = subprocess.Popen(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            close_fds=True,
        )
        try:
            _, _ = p_loaddata.communicate(timeout=timeout)
            ret_code = p_loaddata.poll()
            if ret_code == 0:
                logger.info("load data finished!")
            else:
                logger.info("load benchmark get error {}".format(ret_code))
                return FAILED
        except subprocess.TimeoutExpired:
            # benchmark_timeout = True
            logger.info("load data timeout!")
            return FAILED

    def run_benchmark(self, cmd, filename, collect_resource=False):
        logger.info("benchmark start!")
        benchmark_timeout = False
        p_benchmark = subprocess.Popen(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            close_fds=True,
        )

        # Internal Metrics & Resource Consumption Collection ↓
        # start Internal Metrics Collection
        internal_metrics = Manager().list()
        im = mp.Process(
            target=self.db.get_internal_metrics,
            args=(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME),
        )
        self.db.set_im_alive(True)
        im.start()
        # start Resource Monition (if activated)

        # use docker instead of a real remote instance
        if collect_resource:
            rm = ResourceMonitor(
                self.db.pid,
                1,
                BENCHMARK_WARMING_TIME,
                BENCHMARK_RUNNING_TIME,
                self.remote_mode,
            )
            rm.run()
        # Internal Metrics & Resource Consumption Collection ↑

        try:
            _, _ = p_benchmark.communicate(timeout=600)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                logger.info("benchmark finished!")
            else:
                logger.info("run benchmark get error {}".format(ret_code))
                return
        except subprocess.TimeoutExpired:
            # benchmark_timeout = True
            logger.info("benchmark timeout!")

        # terminate Benchmark
        if not self.remote_mode:
            subprocess.Popen(
                self.db.clear_cmd,
                shell=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                close_fds=True,
            )
            logger.info("clear processlist")

        # stop Internal Metrics Collection
        self.db.set_im_alive(False)
        im.join()

        # stop Resource Monition (if activated)
        if collect_resource:
            rm.terminate()
            (
                cpu,
                avg_read_io,
                avg_write_io,
                avg_virtual_memory,
                avg_physical_memory,
            ) = rm.get_monitor_data_avg()
        else:
            cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory = (
                0,
                0,
                0,
                0,
                0,
            )

        external_metrics = self.get_external_metrics(filename)
        internal_metrics, dirty_pages, hit_ratio, page_data = self.db._post_handle(
            internal_metrics
        )
        logger.info("internal metrics: {}.".format(list(internal_metrics)))

        return (
            benchmark_timeout,
            external_metrics,
            internal_metrics,
            (
                cpu,
                avg_read_io,
                avg_write_io,
                avg_virtual_memory,
                avg_physical_memory,
                dirty_pages,
                hit_ratio,
                page_data,
            ),
        )
    def get_states(self, collect_resource=False):
        # start Benchmark
        tasks = self.args_tune.get("tasks", "None")
        tasks = eval(tasks) if type(tasks) is str else tasks
        weights = []
        if tasks is not None and len(tasks) > 0:
            res = []
            for task in tasks:
                self.db.restart()
                time.sleep(10)
                
                tmp = task.split("_")
                wkld_type = tmp[0]
                wkld_weight = tmp[1:-1]

                weights.append(float(tmp[-1]))
                self.db.dbname = wkld_type
                template_file = os.path.dirname(self.oltpbench_config_xml) + f"/template_{wkld_type}_config.xml"
                with open(
                    template_file,
                    "r",
                ) as f:
                    src = Template(f.read())
                    context = src.substitute({
                        "weights": ",".join(wkld_weight),
                        "warmup": self.args_db["workload_warmup_time"],
                        "time": self.args_db["workload_time"]
                        })
                
                template_file = os.path.dirname(self.oltpbench_config_xml) + f"/tmp/synthesis_{wkld_type}_{'_'.join(wkld_weight)}.xml" 
                with open(template_file, "w") as f:
                    f.write(context)
                cmd, filename = self.get_benchmark_cmd(oltpbench_config_xml=template_file)
                s = self.run_benchmark(cmd, filename, collect_resource)
                res.append(s)

            
            # Post handle metrics collected
            benchmark_timeout = []
            external_metrics = []
            internal_metrics = []
            resource = []
            for i in range(len(res)):
                benchmark_timeout1, external_metrics1, internal_metrics1, resource1 = res[i]
                benchmark_timeout.append(benchmark_timeout1)
                external_metrics.append(external_metrics1)
                internal_metrics.append(internal_metrics1)
                resource.append(resource1)
            benchmark_timeout = True in benchmark_timeout
            external_metrics = np.sum([np.array(external_metrics[i]) * weights[i] for i in range(len(external_metrics))], axis=0).tolist()
            internal_metrics = np.sum([np.array(internal_metrics[i]) * weights[i] for i in range(len(internal_metrics))], axis=0).tolist()
            resource = np.sum([np.array(resource[i]) * weights[i] for i in range(len(resource))], axis=0).tolist()

            return (benchmark_timeout, external_metrics, internal_metrics, resource)
        
        else:
            cmd, filename = self.get_benchmark_cmd()

            return self.run_benchmark(cmd, filename, collect_resource)

    def apply_knobs(self, knobs):
        for key in knobs.keys():
            value = knobs[key]
            if (
                not key in self.knobs_detail.keys()
                or not self.knobs_detail[key]["type"] == "integer"
            ):
                continue
            if value > self.knobs_detail[key]["max"]:
                knobs[key] = self.knobs_detail[key]["max"]
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]["min"]:
                knobs[key] = self.knobs_detail[key]["min"]
                logger.info(
                    "{} with value of is smaller than min, adjusted".format(key)
                )

        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))
        if self.online_mode:
            flag = self.db.apply_knobs_online(knobs)
        else:
            flag = self.db.apply_knobs_offline(knobs)

        if not flag:
            if self.reinit:
                logger.info("reinitializing db begin")
                self.db.reinitdb_magic(self.remote_mode)
                logger.info("db reinitialized")

            raise Exception("Apply knobs failed!")

    def step_GP(self, knobs, collect_resource=False):
        # return False, np.random.rand(6), np.random.rand(65), np.random.rand(8)
        # re-init database if activated
        if self.reinit_interval > 0 and self.reinit_interval % RESTART_FREQUENCY == 0:
            if self.reinit:
                logger.info("reinitializing db begin")
                self.db.reinitdb_magic(self.remote_mode)
                logger.info("db reinitialized")
        self.step_count = self.step_count + 1
        self.reinit_interval = self.reinit_interval + 1

        # modify and apply knobs
        for key in knobs.keys():
            value = knobs[key]
            if (
                not key in self.knobs_detail.keys()
                or not self.knobs_detail[key]["type"] == "integer"
            ):
                continue
            if value > self.knobs_detail[key]["max"]:
                knobs[key] = self.knobs_detail[key]["max"]
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]["min"]:
                knobs[key] = self.knobs_detail[key]["min"]
                logger.info(
                    "{} with value of is smaller than min, adjusted".format(key)
                )

        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))

        # No need to reload data actually

        # apply knobs and start benchmark
        if self.online_mode:
            flag = self.db.apply_knobs_online(knobs)
        else:
            flag = self.db.apply_knobs_offline(knobs)

        if not flag:
            if self.reinit:
                logger.info("reinitializing db begin")
                self.db.reinitdb_magic(self.remote_mode)
                logger.info("db reinitialized")

            raise Exception("Apply knobs failed!")

        s = self.get_states(collect_resource=collect_resource)

        if s is None:
            if self.reinit:
                logger.info("reinitializing db begin")
                self.db.reinitdb_magic(self.remote_mode)
                logger.info("db reinitialized")

            raise Exception("Get states failed!")

        timeout, external_metrics, internal_metrics, resource = s

        format_str = "{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n"
        res = format_str.format(
            knobs,
            str(external_metrics[0]),
            str(external_metrics[1]),
            str(external_metrics[2]),
            external_metrics[3],
            external_metrics[4],
            external_metrics[5],
            resource[0],
            resource[1],
            resource[2],
            resource[3],
            resource[4],
            resource[5],
            resource[6],
            resource[7],
            list(internal_metrics),
        )

        return timeout, external_metrics, internal_metrics, resource

    def get_objs(self, res):
        objs = []
        for y_variable in self.y_variable:
            key = y_variable.strip().strip("-")
            value = res[key]
            if not y_variable.strip()[0] == "-":
                value = -value
            objs.append(value)

        return objs

    def get_constraints(self, res):
        if len(self.constraints) == 0:
            return None

        locals().update(res)
        constraintL = []
        for constraint in self.constraints:
            value = eval(constraint)
            constraintL.append(value)

        return constraintL

    def step(self, config):
        knobs = config.get_dictionary().copy()
        for k in self.knobs_detail.keys():
            if k in knobs.keys():
                if (
                    self.knobs_detail[k]["type"] == "integer"
                    and self.knobs_detail[k]["max"] > sys.maxsize
                ):
                    knobs[k] = knobs[k] * 1000
            else:
                knobs[k] = self.knobs_detail[k]["default"]

        try:
            timeout, metrics, internal_metrics, resource = self.step_GP(
                knobs, collect_resource=True
            )

            if timeout:
                trial_state = TIMEOUT
            else:
                trial_state = SUCCESS

            external_metrics = {
                "tps": metrics[0],
                "lat": metrics[1],
                "qps": metrics[2],
                "tpsVar": metrics[3],
                "latVar": metrics[4],
                "qpsVar": metrics[5],
            }

            resource = {
                "cpu": resource[0],
                "readIO": resource[1],
                "writeIO": resource[2],
                "IO": resource[1] + resource[2],
                "virtualMem": resource[3],
                "physical": resource[4],
                "dirty": resource[5],
                "hit": resource[6],
                "data": resource[7],
            }

            res = dict(external_metrics, **resource)
            objs = self.get_objs(res)
            constraints = self.get_constraints(res)
            return (
                objs,
                constraints,
                external_metrics,
                resource,
                list(internal_metrics),
                self.info,
                trial_state,
            )

        except Exception as e:
            # print(e.with_traceback())
            return None, None, {}, {}, [], self.info, FAILED
