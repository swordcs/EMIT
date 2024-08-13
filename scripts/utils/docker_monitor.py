import time
import docker
import multiprocessing as mp

from multiprocessing import Manager


class DockerMonitor:
    def __init__(self, interval, warmup, t, docker="1"):
        self.interval = interval
        self.warmup = warmup
        self.t = t
        self.alive = mp.Value("b", False)
        self.docker_name = "postgres_gj_" + str(docker)
        self.io_read_seq = Manager().list()
        self.io_write_seq = Manager().list()
        self.cpu_usage_seq = Manager().list()
        self.mem_physical_usage_seq = Manager().list()

        self.processes = []

    def run(self):
        p1 = mp.Process(target=self.monitor_docker, args=())
        self.processes.append(p1)
        [proc.start() for proc in self.processes]
        self.alive.value = True

    def get_monitor_data(self):
        [proc.join() for proc in self.processes]
        return (
            list(self.cpu_usage_seq),
            list(self.mem_physical_usage_seq),
            list(self.io_read_seq),
            list(self.io_write_seq),
        )

    def monitor_docker(self):
        client = docker.from_env()
        container = client.containers.get(self.docker_name)

        start_time = time.time()
        while self.alive.value:
            if time.time() - start_time < self.warmup:
                time.sleep(self.interval)
                continue

            stats1 = container.stats(stream=False)
            time.sleep(self.interval)
            stats2 = container.stats(stream=False)

            cpu_delta = (
                stats1["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats1["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_cpu_delta = (
                stats1["cpu_stats"]["system_cpu_usage"]
                - stats1["precpu_stats"]["system_cpu_usage"]
            )
            number_cpus = stats1["cpu_stats"]["online_cpus"]

            cpu_usage = cpu_delta / system_cpu_delta * number_cpus * 100.0
            self.cpu_usage_seq.append(cpu_usage)

            mem_usage = (
                (
                    stats1["memory_stats"]["usage"]
                    - stats1["memory_stats"]["stats"]["inactive_file"]
                )
                / 1024.0
                / 1024.0
                / 1024.0
            )
            self.mem_physical_usage_seq.append(mem_usage)

            io_stats1 = stats1["blkio_stats"]["io_service_bytes_recursive"]
            io_stats2 = stats2["blkio_stats"]["io_service_bytes_recursive"]
            io_stats1 = {_["op"]: _["value"] / 1024.0 / 1024.0 for _ in io_stats1}
            io_stats2 = {_["op"]: _["value"] / 1024.0 / 1024.0 for _ in io_stats2}
            io_read = io_stats2["read"] - io_stats1["read"]
            io_write = io_stats2["write"] - io_stats1["write"]
            self.io_read_seq.append(io_read)
            self.io_write_seq.append(io_write)

    def terminate(self):
        self.alive.value = False
