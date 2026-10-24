#!/usr/bin/env julia
# Paired finite-CSDA alternative at fixed first-order statistics and exposures.
# The planner waits for and verifies the completed first-order experiment.
using DiffPumas.Tomography
using TOML, SHA, Serialization, SparseArrays, LinearAlgebra, Statistics, Random, DelimitedFiles

function extension_csv(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows;println(io,join(row,','));end
    end
end

function finite_extension(config_path)
    cfg=TOML.parsefile(config_path)
    cfg["exposure_factors"]==[.5,1.,2.,4.,8.,16.] || error("Predeclared exposure grid changed")
    for (path,digest) in cfg["input_sha256"]
        bytes2hex(sha256(read(path)))==digest || error("Frozen input changed: $path")
    end
    out=cfg["output"];ispath(out) && error("Refuse existing output");mkpath(out)
    cp(config_path,joinpath(out,"config.toml"))
    arrays=deserialize(cfg["arrays"])
    f,J,s,lens,roi,profiles=arrays.f,arrays.J,arrays.signal,arrays.lens,arrays.roi,arrays.profiles
    n=length(f);n==2700 && size(J)==(2700,600) && length(lens)==4 || error("Headline shape changed")
    # Directional rows are frozen, audited output, with this exact source header.
    template_path=cfg["directional_templates"]
    header=split(readline(template_path),',')
    header==["bin","theta_deg","phi_deg","omega_sr","dry_flux","linear_signal","finite_lens_flux","roi"] || error("Template schema changed")
    raw_templates=readdlm(template_path,',',Any;skipstart=1)
    size(raw_templates)==(n,8) || error("Template shape changed")
    templates=Float64.(raw_templates[:,1:7])
    string.(raw_templates[:,8])==string.(roi) || error("Frozen ROI/template mismatch")
    templates[:,1]==collect(1:n) || error("Template row order changed")
    all(isequal.(templates[:,5],f)) && all(isequal.(templates[:,6],s)) || error("Frozen template/array mismatch")
    finite_flux=templates[:,7];all(>(0),finite_flux) || error("Nonpositive finite-CSDA rates")
    profiles_csv=readdlm(cfg["exposure_shares"],',',Float64;skipstart=1)
    expected_names=["legacy_equal_bin","uniform_directional_area","cosine","azimuth_modulated"]
    [name for (name,_) in profiles]==expected_names || error("Profile definition/order changed")
    for (i,(_,p)) in enumerate(profiles)
        all(isequal.(profiles_csv[:,i+1],p)) && isapprox(sum(p),1;atol=1e-14) || error("Frozen exposure profile mismatch")
    end
    amplitude(A,y)=mean(first(sart_reconstruct(A,y;n_iter=200,relaxation=.2))[lens])
    seed=Int(cfg["seed"]);alpha=.0013498980316301
    helper=cfg["poisson_helper"];rows=Any[];replay=Any[]
    cached=Dict{Tuple{String,Int,String},Matrix{Float64}}()
    # Finish every source replay gate before generating any new population.
    for (name,p) in profiles
        b=p.*f;d=p.*s;B=spdiagm(0=>p)*J;Epsi=4.645^2/sum(d.^2 ./ b)
        for (fi,factor) in enumerate(cfg["exposure_factors"][1:3])
            E=factor*Epsi
            for (kind,draws) in (("calibration_null",3000),("test_null",3000),("linear_signal",600))
                original_kind=kind=="linear_signal" ? "signal" : kind
                old=joinpath(cfg["firstorder_results"],name*"_poisson_"*string(factor))
                values=readdlm(joinpath(old,original_kind*"_statistics.csv"),',',Float64)
                size(values)==(draws,3) || error("Original statistic shape mismatch")
                counts=reshape(reinterpret(Int64,read(joinpath(old,original_kind*".bin"))),n,draws)
                for i in unique([1,draws÷2,draws])
                    excess=counts[:,i]./E.-b
                    check=[dot(d./b,excess),sum(excess[roi]),amplitude(B,excess)]
                    all(isapprox.(check,values[i,:];rtol=1e-12,atol=1e-12)) || error("Frozen statistic replay mismatch")
                    push!(replay,(name,factor,kind,i,maximum(abs.(check.-values[i,:]))))
                end
                cached[(name,fi,kind)]=values
            end
        end
    end
    length(replay)==108 || error("Source replay gate incomplete")
    extension_csv(joinpath(out,"reused_statistic_replay.csv"),["profile","factor","population","draw","max_abs_difference"],replay)
    header_out=["profile","method","hypothesis","exposure_factor","total_E","expected_null_counts","expected_signal_counts","threshold","null_exceedances","null_draws","signal_exceedances","signal_draws","null_source"]
    for (pi,(name,p)) in enumerate(profiles)
        b=p.*f;d=p.*s;B=spdiagm(0=>p)*J;Epsi=4.645^2/sum(d.^2 ./ b)
        for (fi,factor) in enumerate(cfg["exposure_factors"])
            println("Finite-alternative paired pilot: profile=$name factor=$factor");flush(stdout)
            E=factor*Epsi;folder=joinpath(out,name*"_poisson_"*string(factor));mkpath(folder)
            stats=Dict{String,Matrix{Float64}}(); source_kind=fi<=3 ? "reused_clean_firstorder" : "fresh_paired"
            for (ki,(kind,draws,rate)) in enumerate((("calibration_null",3000,b),("test_null",3000,b),("linear_signal",600,b+d),("finite_signal",600,p.*finite_flux)))
                if fi<=3 && ki<=3
                    stats[kind]=cached[(name,fi,kind)]
                else
                    rate_path=joinpath(folder,kind*"_rates.txt");writedlm(rate_path,rate)
                    count_path=joinpath(folder,kind*".bin")
                    local_seed=seed+1000000*pi+10000*fi+1000*ki
                    run(`python3 $helper draw --rates $rate_path --exposure $E --draws $draws --seed $local_seed --output $count_path`)
                    counts=reshape(reinterpret(Int64,read(count_path)),n,draws);values=zeros(draws,3)
                    Threads.@threads for i in 1:draws
                        excess=counts[:,i]./E.-b
                        values[i,1]=dot(d./b,excess);values[i,2]=sum(excess[roi]);values[i,3]=amplitude(B,excess)
                    end
                    stats[kind]=values;writedlm(joinpath(folder,kind*"_statistics.csv"),values,',')
                end
            end
            for (mi,method) in enumerate(("matched_first_order","roi_count","sart_200"))
                cut=quantile(stats["calibration_null"][:,mi],1-alpha)
                kn=count(>(cut),stats["test_null"][:,mi])
                for (hypothesis,kind,rate) in (("first_order","linear_signal",b+d),("finite_csda","finite_signal",p.*finite_flux))
                    ks=count(>(cut),stats[kind][:,mi])
                    push!(rows,(name,method,hypothesis,factor,E,E*sum(b),E*sum(rate),cut,kn,3000,ks,600,source_kind))
                end
            end
            raw=joinpath(out,"paired_pilot_raw.csv");extension_csv(raw,header_out,rows)
            run(`python3 $helper intervals --input $raw --output $(joinpath(out,"paired_pilot_intervals.csv"))`)
            extension_csv(joinpath(out,"reused_statistic_replay.csv"),["profile","factor","population","draw","max_abs_difference"],replay)
        end
    end
    open(joinpath(out,"scope.toml"),"w") do io
        TOML.print(io,Dict("scope"=>"Same fixed first-order matched/ROI/SART statistics, shared held-out nulls and thresholds, two predeclared signal hypotheses. Finite alternative is the biased deterministic CSDA surrogate, not stochastic transport truth or validated physical finite-contrast power.",
            "exposure"=>"Same E/Epsi_linear=.5,1,2,4,8,16 for both hypotheses. Four fixed acceptance profiles at fixed total integrated exposure; expected counts separately reported.",
            "reuse"=>"First three exposure factors reuse clean hash-verified first-order null calibration/test and signal statistics; 108 source draws replay operator/statistics to 1e-12 absolute/relative tolerance. New finite signals use distinct stream kind4. Higher exposures use fresh paired null calibration/test and both signal populations.",
            "intervals"=>"Exact binomial Monte Carlo intervals conditional on realized calibration threshold; threshold uncertainty not propagated. 3000 nulls imply only about4 nominal3sigma-tail events; no3sigma certification."))
    end
end

length(ARGS)==1 || error("Expected one frozen extension config or --load-only")
if ARGS[1]=="--load-only"
    println("Finite acceptance extension runtime include/load smoke: PASS")
else
    finite_extension(abspath(ARGS[1]))
end
