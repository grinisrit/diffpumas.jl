using Test, DiffPumas, DiffPumas.Tomography, Zygote

@testset "Material-conditioned CSDA correction" begin
    T = DiffPumas.Tomography
    mat = MaterialConfig(1, 2, 3, -1, 2650., 1000., 0., 0.)
    path = DirectionalPath(30., 0., [CellSegment(1,30.),CellSegment(2,70.)],100.,200.,100.,0.,true)
    shallow = falses(2)
    w = [0.2,0.4]
    old = get_csda_material_correction()
    try
        set_csda_material_correction!(linear=-0.7,quadratic=0.3)
        # Independent column calculation includes dry fixed remainder.
        gx = [30.,70.] ./ ((1 .- w)./2650 .+ w./1000)
        q = sum(w.*gx)/(sum(gx)+100*2650)
        @test T.csda_material_multiplier(path,shallow,mat,w) ≈ exp(-.7*q+.3*q*q)
        @test T.csda_material_multiplier(path,shallow,mat,zeros(2)) == 1.
        f(x) = T.csda_material_multiplier(path,shallow,mat,x)
        ad = Zygote.gradient(f,w)[1]
        h = 1e-6
        fd = [(f(w+[i==j ? h : 0. for i in 1:2])-f(w-[i==j ? h : 0. for i in 1:2]))/(2h) for j in 1:2]
        @test ad ≈ fd rtol=1e-7
        # At the physical dry boundary, use a one-sided difference.
        dryad = Zygote.gradient(f,zeros(2))[1]
        @test dryad[1] ≈ (f([h,0.])-f(zeros(2)))/h rtol=1e-5
        physics = create_physics(MUON; n_energies=40, K_min=0.001, K_max=1e6,
            materials=[STANDARD_ROCK, AIR, WATER])
        rock = get_material_index(physics,"StandardRock")
        water = get_material_index(physics,"Water")
        air = get_material_index(physics,"Air")
        actual = MaterialConfig(rock,water,air,-1,Float64(physics.tables[rock].density),
            Float64(physics.tables[water].density),0.,0.)
        site = SiteConfig(1000.,12000.)
        samples = csda_energy_quadrature(12,1.,1e5)
        fv, ids, grads = directional_flux_and_grad_csda(physics,shallow,actual,site,path,w,samples)
        full(x) = compute_directional_flux_csda(physics,shallow,actual,site,path,x,samples)
        @test fv ≈ full(w)
        @test grads ≈ Zygote.gradient(full,w)[1][ids] rtol=1e-7
        @test grads ≈ [(full(w+[i==j ? h : 0. for i in 1:2])-full(w-[i==j ? h : 0. for i in 1:2]))/(2h) for j in ids] rtol=1e-4
        fields = [[.1,.1],[.5,.5],[.8,.3]]
        base = zeros(2)
        # Synthetic contrast test checks API algebra, not MC predictive validity.
        m0 = [full(base)]
        mf = reshape([full(x) for x in fields],:,1)
        fit = calibrate_csda_material_correction(physics,shallow,actual,site,[path],
            base,fields,samples,m0,mf,[1];ridge=0.)
        @test fit.linear ≈ -.7 atol=1e-9
        @test fit.quadratic ≈ .3 atol=1e-9
        @test fit.rms_log_after < 1e-12
        mktemp() do artifact, io
            close(io)
            save_csda_corrections(artifact;metadata=Dict("training_seeds"=>[12,34],"source_hash"=>"fixture"))
            set_csda_material_correction!(enabled=false)
            metadata = load_csda_corrections!(artifact)
            @test get_csda_material_correction().linear ≈ -.7 atol=1e-9
            @test metadata["training_seeds"] == [12,34]
            @test metadata["source_hash"] == "fixture"
        end
        set_csda_material_correction!(enabled=false)
        @test f(w) == 1.
        @test_throws ArgumentError set_csda_material_correction!(linear=NaN)
    finally
        set_csda_material_correction!(enabled=old.enabled,linear=old.linear,quadratic=old.quadratic)
    end
end
