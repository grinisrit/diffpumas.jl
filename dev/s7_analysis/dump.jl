# Exploration: list the keys of one saved reconstruction (test/recs/*.jls).
# usage: julia dump.jl [FILE]   (env S7_OUT = run output dir; default FILE = a GN-0 reconstruction of arm Mc, exposure 7, realisation 1)
using Serialization
O = rstrip(get(ENV, "S7_OUT", "/mnt/big_data/agent-builds/diffpumas-s7-out"), '/') * "/"
f = length(ARGS) >= 1 ? ARGS[1] : O * "test/recs/test_obs_cap20_test_Mc_e7_r1_GN-0_0p1.jls"
d = deserialize(f); println(typeof(d)); println(sort(collect(keys(d)))); println(length(first(values(d))))
