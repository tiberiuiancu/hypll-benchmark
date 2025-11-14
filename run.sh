# only FC, no memory optimizations
# REF="config/triton" BENCH_CONFIGS="triton" python bench_mlp.py -a

# only FC
# REF="config/memory" BENCH_CONFIGS="triton" python bench_mlp.py -a

# fuse relu and logmap, but separately
REF="config/fused-relu-logmap0" BENCH_CONFIGS="triton" python bench_mlp.py -a

# fully fused relu
# REF="config/fused-op-in-tangent-space" BENCH_CONFIGS="triton" python bench_mlp.py -a

# all optimizations on main branch
# REF="main" BENCH_CONFIGS="triton torch euclidean" python bench_mlp.py -a
