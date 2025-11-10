import pstats, cProfile



import self_play


cProfile.runctx("self_play.main()", globals(), locals(), "profile/result_0.prof")

s = pstats.Stats("profile/result_0.prof")
s.strip_dirs().sort_stats("cumtime").print_stats(100)


# --- Setting ---
# Start from scratch
# pre-compile jit functions
# ITERATIONS = 1
# PARALLEL_PLAY = 32
# MCTS_ITERATION = 200/50
# EPOCHS = 100 (with 128 batch size)
