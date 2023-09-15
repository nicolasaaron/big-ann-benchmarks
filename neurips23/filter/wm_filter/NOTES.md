To run locally without the container you can use the `build.local.sh` to build the library locally
Then you can use:

`PYTHONPATH=/home/alessandro/workspace/faiss/build/faiss/python/build/lib/:/home/alessandro/workspace/big-ann-benchmarks/neurips23/filter/wm_filter/ python run.py --neurips23track filter    --algorithm wm_filter   --dataset yfcc-10M --nodocker`

The `PYTHONPATH` has to include both the Faiss library path as well the as the `wm_filter` folder where the library was build by `build_local.sh`