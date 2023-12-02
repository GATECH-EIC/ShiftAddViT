# Shift/Add operators on TVM

Test TVM-based shift/add operators on computed shapes that appear in PVT

## Build up TVM:

Please follow [TVM documentation](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github) to compile.  "set(USE_CUDA ON)" is necessary

Since we need to  auto-tuning module, Python dependencies below is needed.

```bash
pip install tornado psutil 'xgboost>=1.1.0' cloudpickle
```



## Shift op:

Please move to the corresponding folder first.

```bash
cd ./OPs_Speedups/Shift
```

1. build op and tune:

```bash
python Ansor_tune.py
```

2. build op and test latency

```bash
python latency_test.py
```



## Add op:

Please move to the corresponding folder first.

```bash
cd ./OPs_Speedups/Add
```

1. build op and tune:

```bash
python Ansor_tune.py
```

2. build op and test latency

```bash
python latency_test.py
```

