# Exploration: size and keys of the exact-CSDA test template.   usage: julia tl.jl   (env S7_OUT)
using Serialization
O = rstrip(get(ENV, "S7_OUT", "/mnt/big_data/agent-builds/diffpumas-s7-out"), '/') * "/"
c=deserialize(O*"templates/template_test_X_h0_s0.jls"); println(length(c.template), " ", keys(c))
