import pstats, cProfile



import self_play


# cProfile.runctx("self_play.main()", globals(), locals(), "profile/result_0.prof")

s = pstats.Stats("profile/result_0.prof")
s.strip_dirs().sort_stats("cumtime").print_stats(100)
