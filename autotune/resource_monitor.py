import time
import psutil
import docker
import multiprocessing as mp
from multiprocessing import Manager

class ResourceMonitor:
    def __init__(self, pid, interval, warmup, t, remote_mode=False):
        self.interval = interval
        self.warmup = warmup
        self.t = t
        self.n_cpu = 4
        self.remote_mode = remote_mode
        self.ticks = int(self.t / self.interval)
        self.processes = []
        self.alive = mp.Value("b", False)
        self.docker_name = "postgres_gj_1"
        if not remote_mode:
            self.process = psutil.Process(pid)
        self.io_read_seq = Manager().list()
        self.io_write_seq = Manager().list()
        self.cpu_usage_seq = Manager().list()
        self.dirty_pages_pct_seq = Manager().list()
        self.mem_virtual_usage_seq = Manager().list()
        self.mem_physical_usage_seq = Manager().list()

    def run(self):
        if self.remote_mode:
            p1 = mp.Process(target=self.monitor_docker, args=())
            self.processes.append(p1)
        else:
            self.n_cpu = len(self.process.cpu_affinity())

            p1 = mp.Process(target=self.monitor_cpu_usage)
            self.processes.append(p1)
            p2 = mp.Process(target=self.monitor_mem_usage)
            self.processes.append(p2)
            p3 = mp.Process(target=self.monitor_io_usage)
            self.processes.append(p3)

        [proc.start() for proc in self.processes]
        self.alive.value = True

    def get_monitor_data(self):
        [proc.join() for proc in self.processes]
        return {
            "mem_virtual": list(self.mem_virtual_usage_seq),
            "mem_physical": list(self.mem_physical_usage_seq),
            "io_read": list(self.io_read_seq),
            "io_write": list(self.io_write_seq),
        }

    def get_monitor_data_avg(self):
        [proc.join() for proc in self.processes]
        cpu = list(self.cpu_usage_seq)
        mem_virtual = list(self.mem_virtual_usage_seq)
        mem_physical = list(self.mem_physical_usage_seq)
        io_read = list(self.io_read_seq)
        io_write = list(self.io_write_seq)

        avg_cpu = sum(cpu) / (len(cpu) + 1e-9) / self.n_cpu
        avg_read_io = sum(io_read) / (len(io_read) + 1e-9)
        avg_write_io = sum(io_write) / (len(io_write) + 1e-9)
        avg_virtual_memory = sum(mem_virtual) / (len(mem_virtual) + 1e-9)
        avg_physical_memory = sum(mem_physical) / (len(mem_physical) + 1e-9)
        return (
            avg_cpu,
            avg_read_io,
            avg_write_io,
            avg_virtual_memory,
            avg_physical_memory,
        )

    def monitor_mem_usage(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            try:
                tmp_processes = self.get_all_process()
                mem_physical = sum(
                    [
                        proc.memory_info()[0] / (1024.0 * 1024.0 * 1024.0)
                        for proc in tmp_processes
                    ]
                )
                mem_virtual = sum(
                    [
                        proc.memory_info()[1] / (1024.0 * 1024.0 * 1024.0)
                        for proc in tmp_processes
                    ]
                )
                self.mem_physical_usage_seq.append(mem_physical)
                self.mem_virtual_usage_seq.append(mem_virtual)
                time.sleep(self.interval)
                count += 1
            except psutil.NoSuchProcess:
                pass

    def monitor_io_usage(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            tmp_processes = self.get_all_process()
            sp1 = dict()
            sp2 = dict()
            for proc in tmp_processes:
                try:
                    sp1[proc] = proc.io_counters()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            time.sleep(self.interval)
            tmp_processes = self.get_all_process()
            for proc in tmp_processes:
                try:
                    sp2[proc] = proc.io_counters()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            tmp_io_read = 0
            tmp_io_write = 0

            for proc in sp2.keys():
                if proc in sp1.keys():
                    tmp_io_read += sp2[proc][2] - sp1[proc][2]
                    tmp_io_write += sp2[proc][3] - sp1[proc][3]
                else:
                    tmp_io_read += sp2[proc][2]
                    tmp_io_write += sp2[proc][3]
            self.io_read_seq.append(tmp_io_read / (1024.0 * 1024.0))
            self.io_write_seq.append(tmp_io_write / (1024.0 * 1024.0))
            count += 1

    def monitor_cpu_usage(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            tmp_processes = self.get_all_process()
            try:
                cpu = 0
                for proc in tmp_processes:
                    if psutil.pid_exists(proc.pid):
                        cpu += proc.cpu_percent(
                            interval=max(self.interval / len(tmp_processes), 0.01)
                        )
                self.cpu_usage_seq.append(cpu)
            except psutil.NoSuchProcess:
                pass
            count += 1

    def monitor_docker(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            client = docker.from_env()
            container = client.containers.get(self.docker_name)
            stats1 = container.stats(stream=False)
            time.sleep(self.interval)
            stats2 = container.stats(stream=False)
            cpu_usage = (
                stats1["cpu_stats"]["cpu_usage"]["total_usage"]
                / stats1["cpu_stats"]["system_cpu_usage"]
                * 100.0
            )
            self.cpu_usage_seq.append(cpu_usage)

            mem_usage = stats1["memory_stats"]["usage"] / 1024.0 / 1024.0 / 1024.0
            self.mem_physical_usage_seq.append(mem_usage)
            self.mem_virtual_usage_seq.append(0.0)

            io_stats1 = stats1["blkio_stats"]["io_service_bytes_recursive"]
            io_stats2 = stats2["blkio_stats"]["io_service_bytes_recursive"]
            io_stats1 = {_["op"]: _["value"] / 1024.0 / 1024.0 for _ in io_stats1}
            io_stats2 = {_["op"]: _["value"] / 1024.0 / 1024.0 for _ in io_stats2}
            io_read = io_stats2["read"] - io_stats1["read"]
            io_write = io_stats2["write"] - io_stats1["write"]
            self.io_read_seq.append(io_read)
            self.io_write_seq.append(io_write)
            count = count + 1


    def terminate(self):
        self.alive.value = False

    def get_all_process(self):
        return [self.process] + self.process.children()
