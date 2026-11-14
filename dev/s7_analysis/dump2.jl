# Writes dump.tsv: one line per "file#key" of need.txt (made by mklist.py) with the saved reconstruction vector.
# usage: julia dump2.jl   (env S7_OUT = run output dir, S7_WORK = work dir with need.txt; writes S7_WORK/dump.tsv)
using Serialization
O = rstrip(get(ENV, "S7_OUT", "/mnt/big_data/agent-builds/diffpumas-s7-out"), '/') * "/"
W = rstrip(get(ENV, "S7_WORK", "."), '/') * "/"
cache=Dict{String,Any}()
open(W*"dump.tsv","w") do io
  for l in eachline(W*"need.txt")
    f,lbl=split(l,'#'; limit=2)
    d=get!(cache,f) do; deserialize(O*"test/"*f); end
    v=d[lbl]; println(io,l,'\t',join(string.(v),','))
  end
end
