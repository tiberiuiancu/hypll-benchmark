# only FC
# REF="memory" BENCH_CONFIGS="triton" python bench_mlp.py -a

# fuse relu and logmap, but separately
# REF="fused-relu-logmap0" BENCH_CONFIGS="triton" python bench_mlp.py -a

# fully fused relu
# REF="fc-bwd-1d-grid" BENCH_CONFIGS="triton" python bench_mlp.py -a

# only calculate dx if input requires grad
REF="main" BENCH_CONFIGS="triton torch euclidean" python bench_mlp.py -a
