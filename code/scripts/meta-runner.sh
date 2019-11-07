#!/usr/bin/env bash
#for seed in 12 26 100 365; do
#for seed in  500 1024 3423; do
#for seed in  500 1024 3423 32542 100000; do
#for temperature in 1 2 5 10 15 20 25 50 75 100; do
#0.0
# 0.001 0.0003 0.003 0.005; do
# 0.25 0.5 0.75 0.9
for seed in 1 12 26 100 365; do
    for temperature in  1.0 1.5 2.0; do
        for kdalpha in  0.75 0.9; do
            for lr in 0.003;  do
				for lang in hi zh; do
#    sbatch train_integrated_xlingual_transfer.sh ${seed}
               sbatch train_integrated_xlingual_st.sh ${seed} ${temperature} ${kdalpha} ${lr} ${lang}
#    sbatch train_integrated_xlingual_translation_baseline.sh ${seed}
            done
        done
    done
done
