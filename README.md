# This is the code repository for EMIT

## Usage

1. Create conda environment first as follows:

```shell
conda create -n emit python=3.8
conda activate emit

pip install -r ./requirements.txt
```

2. Create postgresql database instance through Docker:

```shell
docker run -m 8G --cpus=4 -it -d --name postgres -p 10022:22 -p 15432:5432 gj9650/ubuntu-postgres 
```

3. Setup the .ini file with corresponding information (ssh port, private_key and etc.)

- For example, if the objective of tuning is tps, the file can be set as `performance_metric = ['tps']`.
- The tuning method is configured by `optimize_method`.
- The acquisition function is configured by `acq_type`, `eit` is where our method implemented.

4. Run the database tuning program:

```shell
python ./scripts/optimize.py python --config scripts/ini/config.ini 
```

**P.S. We use [benchbase](https://github.com/cmu-db/benchbase) as the benchmarking tool, so you may need to install it locally to support the tuning.**