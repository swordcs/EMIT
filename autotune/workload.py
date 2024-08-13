from dynaconf import settings
# NUM_TABLES = 16
# TABLE_RANGE = 200000


OLTPBENCH_WORKLOADS = {
    'name': 'oltpbench',
    'type': 'oltpbenchmark',
    # bash run_oltpbench.sh benchmark config_xml output_file
    'cmd': 'bash {} {} {} {} {} {} {}' 
}
