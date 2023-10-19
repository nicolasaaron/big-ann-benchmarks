#!/bin/sh

python run.py --neurips23track filter    --algorithm wm_filter   --dataset yfcc-10M

python run.py --neurips23track filter    --algorithm wm_filter_direct_1k   --dataset yfcc-10M

python run.py --neurips23track filter    --algorithm wm_filter_direct_2k   --dataset yfcc-10M

python run.py --neurips23track filter    --algorithm wm_filter_direct_1k_pq_refine   --dataset yfcc-10M

python run.py --neurips23track filter    --algorithm wm_filter_direct_2k_pq_refine   --dataset yfcc-10M

sudo chmod 777 -R results/
python plot.py --dataset yfcc-10M --neurips23track filter
python data_export.py --out res.csv


