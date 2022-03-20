python main.py -d "data/real/raw" \
            -n unified_label_toy_cami_hmp \
            -l 30 \
            --n_shared_read 5 \
            --seed_size 30 \
            --n_clusters 35 \
            --result_dir result \
            --is_amd \
            --load_feature_cache \
            --load_graph_cache \
            --graph_cache "data/real/processed/unified_label_toy_cami_hmp.fna.groups.json" \
            --kfeature_cache "data/real/processed/unified_label_toy_cami_hmp" \
            