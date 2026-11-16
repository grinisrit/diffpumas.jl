# Writes tmpl.tsv (templates of the test field: X, dev X, M replicates 431-438 and their sigma_mc) for item3.py.
# usage: julia tl2.jl   (env S7_OUT = run output dir, S7_WORK = work dir)
using Serialization
O = rstrip(get(ENV, "S7_OUT", "/mnt/big_data/agent-builds/diffpumas-s7-out"), '/') * "/"
W = rstrip(get(ENV, "S7_WORK", "."), '/') * "/"
T=O*"templates/"
c=deserialize(T*"template_test_M_h16384_s431.jls"); println(typeof(c.extra)); println(c.extra isa NamedTuple ? keys(c.extra) : (c.extra isa AbstractDict ? keys(c.extra) : size(c.extra)))
println(c.key)
open(W*"tmpl.tsv","w") do io
 for (nm,f) in [("X","template_test_X_h0_s0.jls"),("D_X","template_dev_X_h0_s0.jls")]
   d=deserialize(T*f); println(io,nm,'\t',join(string.(d.template),','))
 end
 for b in 431:438
   d=deserialize(T*"template_test_M_h16384_s$(b).jls")
   println(io,"M$b",'\t',join(string.(d.template),','))
   ex=d.extra
   sg = ex isa NamedTuple && haskey(ex,:sigma_mc) ? ex.sigma_mc : (ex isa AbstractDict && haskey(ex,"sigma_mc") ? ex["sigma_mc"] : nothing)
   sg===nothing || println(io,"S$b",'\t',join(string.(sg),','))
 end
end
