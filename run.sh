REF="triton" BENCH_CONFIGS="torch triton euclidean" python bench_mlp.py -a
REF="memory" BENCH_CONFIGS="triton" python bench_mlp.py -a
# REF="logmap0" BENCH_CONFIGS="triton" python bench_mlp.py -a
# REF="expmap0" BENCH_CONFIGS="triton" python bench_mlp.py -a
REF="fused-relu-logmap0" BENCH_CONFIGS="triton" python bench_mlp.py -a
# REF="fused-op-in-tangent-space" BENCH_CONFIGS="triton" python bench_mlp.py -a
# REF="fc-fwd-1d-grid" BENCH_CONFIGS="triton" python bench_mlp.py -a
REF="fc-bwd-1d-grid" BENCH_CONFIGS="triton" python bench_mlp.py -a
