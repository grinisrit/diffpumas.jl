using Test, Random
using DiffPumas
using DiffPumas.MixedTransport
using DiffPumas.Constants
using DiffPumas.Tomography
@testset "Prepared mixed transport limiting cases" begin
    energies=[1.,1e5]; stopping=[2.,2.]; rates=[0.,0.]
    c=MixedKernel(MUON_MASS,MUON_C_TAU,.05,energies,stopping,rates,
        [0.,(1e5-1)/2],[0.,0.],[0.,0.],(),Inf)
    s=State{Float64}(energy=10.,charge=1.)
    a=transport_mixed_step(c,s,1000.,.01,MersenneTwister(1))
    b=transport_mixed_step(c,s,1000.,.01,MersenneTwister(2))
    @test a.energy ≈ 30.
    @test a.energy == b.energy
    @test a.distance ≈ .01
    @test a.grammage ≈ 10.
    @test a.weight == 1.
    first=transport_mixed_step(c,s,1000.,.003,MersenneTwister(3))
    split=transport_mixed_step(c,first,1000.,.007,MersenneTwister(4))
    @test split.energy ≈ a.energy
    @test split.distance ≈ a.distance
    @test split.grammage ≈ a.grammage
    limited=transport_mixed_step(c,s,1000.,.01,MersenneTwister(1);energy_limit=20.)
    @test limited.energy ≈ 20.
    @test limited.distance ≈ .005
    @test_throws DomainError transport_mixed_step(c,s,1000.,1000.,MersenneTwister(1))
end

mutable struct MixedScriptedRNG <: AbstractRNG
    uniforms::Vector{Float64}
end
Random.rand(rng::MixedScriptedRNG)=popfirst!(rng.uniforms)
@testset "Adjoint hard events and optical survival" begin
    e=(Z=11.,A=22.,I=149e-9,fraction=1.)
    c=MixedKernel(MUON_MASS,MUON_C_TAU,.05,[.001,1e9],[2.,2.],[.2,.2],
        [0.,5e8],[0.,1e8],[0.,0.],(e,),.001)
    ancestor,weight=reverse_del_event(c,1000.,MixedScriptedRNG([.5]))
    nu=1/(1/.05-.5*(1/.05-1))
    @test ancestor ≈ 1000/(1-nu)
    q=ancestor-1000
    raw=DiffPumas.Materials.electronic_dcs(11.,149e-9,MUON_MASS,ancestor,q)+
        DiffPumas.Materials.dcs_bremsstrahlung_ssr(11.,22.,MUON_MASS,ancestor,q)+
        DiffPumas.Materials.dcs_pair_production_ssr(11.,22.,MUON_MASS,ancestor,q)+
        DiffPumas.Materials.dcs_photonuclear_drss(11.,22.,MUON_MASS,ancestor,q)
    pdf=1/((1/.05-1)*nu^2)
    @test weight ≈ ancestor*raw*AVOGADRO_NUMBER/(22*.001)/(pdf*(1-nu)*.2)
    @test weight>0
    # Low-energy null closure: impossible DEL proposal stays unchanged.
    low=MixedKernel(c.mass,c.ctau,c.cutoff,c.energies,c.stopping,c.sigma,
        c.range,c.hazard,c.proper,c.elements,100.)
    @test reverse_del_event(low,.001,MixedScriptedRNG([.999999])) == (.001,1.)
    # A zero differential kernel absorbs at the first proposed hard event;
    # surviving weight-one histories directly measure exp(-sigma*grammage).
    synthetic=MixedKernel(c.mass,c.ctau,c.cutoff,[1.,1e5],[2.,2.],[.1,.1],
        [0.,(1e5-1)/2],[0.,(1e5-1)*.05],[0.,0.],(),1.)
    initial=State{Float64}(energy=10.,charge=1.)
    rng=MersenneTwister(12345)
    survivors=0
    for _ in 1:6000
        final=transport_mixed_step(synthetic,initial,1000.,.01,rng;energy_limit=1e5)
        survivors += final.weight>0
    end
    @test abs(survivors/6000-exp(-1))<.025
end
Random.randexp(rng::MixedScriptedRNG)=popfirst!(rng.uniforms)
@testset "Multiple hard events consume the complete segment" begin
    element=(Z=11.,A=22.,I=149e-9,fraction=1.)
    c=MixedKernel(MUON_MASS,MUON_C_TAU,.05,[1.,1e9],[2.,2.],[.1,.1],
        [0.,(1e9-1)/2],[0.,(1e9-1)*.05],[0.,0.],(element,),1.)
    s=State{Float64}(energy=1000.,charge=1.)
    # Two optical depths produce vertices at grammages1 and3. The final draw
    # places the next vertex beyond the segment, leaving7kg/m² to transport.
    rng=MixedScriptedRNG([.1,.5,.2,.5,10.])
    final=transport_mixed_step(c,s,1000.,.01,rng;energy_limit=1e9)
    r=1-1/(1/.05-.5*(1/.05-1))
    expected=((1000+2)/r+4)/r+14
    @test isempty(rng.uniforms)
    @test final.energy ≈ expected rtol=1e-12
    @test final.distance ≈ .01 rtol=1e-12
    @test final.grammage ≈ 10. rtol=1e-12
    @test final.position ≈ s.position-.01*s.direction
    @test isfinite(final.weight) && final.weight>0
end

@testset "Fixed-path MC scope and segment conservation" begin
    p=create_physics(MUON;n_energies=32,K_min=.001,K_max=1e6,
        materials=[STANDARD_ROCK,AIR,WATER],settings=PhysicsSettings(dedx_path="/nonexistent"))
    rock=get_material_index(p,"StandardRock");air=get_material_index(p,"Air");water=get_material_index(p,"Water")
    cfg=MaterialConfig(rock,water,air,-1,2650.,1000.,0.,0.)
    site=SiteConfig(1000.,12000.)
    # A fixed path is authoritative even if its recorded surface height differs
    # from reconstructing z using nominal zenith and segment distances.
    path=DirectionalPath(20.,0.,[CellSegment(1,100.)],100.,100.,0.,60.,true)
    mats=[MaterialMixture(rock)];dens=[2650.]
    samples=[EnergySample(1000.,1.,1.)]
    @test_throws ArgumentError compute_directional_flux_mc(p,cfg,site,path,mats,dens,samples,42;
        straggling=true,energy_threshold_low=100.,scattering=true)
    flux,sigma=compute_directional_flux_mc(p,cfg,site,path,mats,dens,samples,42;
        straggling=false,energy_threshold_low=100.,scattering=false)
    @test flux>0 && isfinite(flux) && sigma==0
    state=DiffPumas.Tomography.initial_state(99.,1.,20.,0.)
    kernel=prepare_mixed_kernel(p,rock)
    final=DiffPumas.Tomography.propagate_mc_uniform_segment(p,state,MaterialMixture(rock),2650.,100.,MersenneTwister(17);
        straggling=true,energy_threshold_low=100.,kernel_context=kernel)
    @test final.distance ≈ 100. atol=1e-8
    @test final.grammage ≈ 265000. atol=1e-5
    @test_throws ArgumentError DiffPumas.Geometry.compute_flux_single(p,
        TwoLayerGeometry{Float64}(100.,2650.,rock,air),1000.,90.,1.;source_energy_max=1e9)
end

@testset "Binding source-energy ceiling and CSDA surrogate" begin
    p=create_physics(MUON;n_energies=24,K_min=.001,K_max=1e6,
        settings=PhysicsSettings(dedx_path="/nonexistent"))
    flux,grad=compute_flux_gradient(p,2650.,100.,90.,1000.,1.;source_energy_max=10.)
    @test flux==0 && grad==0
    samples=[(1000.,1.,2.),(1000.,-1.,2.)]
    sampled,density_gradient=DiffPumas.Geometry._sampled_csda_flux_gradient(
        p,2650.,100.,90.,samples;source_energy_max=10.)
    @test sampled==0 && density_gradient==0
    result=run_backward_mc(p;rock_thickness=100.,elevation=90.,energy_min=1000.,
        n_samples=2,compute_gradient=true,straggling=false,scattering=false,source_energy_max=10.)
    @test result.flux==result.gradient_flux==0
    @test result.gradient==0
    @test result.source_energy_max==10.
    geometry=TwoLayerGeometry{Float64}(100.,2650.,1,2)
    state=State{Float64}(energy=1.,charge=1.,direction=Vec3{Float64}(0,0,-1))
    final=DiffPumas.Geometry.transport_backward_through_geometry(p,geometry,state,2.;
        rng=MersenneTwister(17),straggling=true,scattering=false,energy_threshold_low=100.)
    @test final.energy>=2.
    @test final.position[3]<100.
end

@testset "Sub-ULP range increments preserve spatial transport" begin
    # Energy is large enough that a small grammage increment disappears when
    # rounded through inverse range and back, while the spatial step is resolved.
    e0=9.690953602732087e8
    c=MixedKernel(MUON_MASS,MUON_C_TAU,.05,[1.,1e9],[1e-6,1e-6],[0.,0.],
        [0.,(1e9-1)*1e6],[0.,0.],[0.,1e8],(),Inf)
    distance=7.821108738426119e-8;density=.09128485154324607
    state=State{Float64}(energy=e0,charge=1.,distance=29999.999899921786)
    range=DiffPumas.MixedTransport.lookup(e0,c.energies,c.range)
    @test range+distance*density == range
    result=transport_mixed_step(c,state,density,distance,MersenneTwister(42);energy_limit=1e9)
    @test result.distance-state.distance ≈ distance atol=4eps(state.distance)
    @test result.grammage ≈ distance*density rtol=1e-12
    @test result.energy>=state.energy
    @test result.time>state.time
    @test result.weight>0
end

@testset "Zero spatial increment at mode changes and killed events" begin
    p=create_physics(MUON;n_energies=12,K_min=1.,K_max=1e6,
        settings=PhysicsSettings(dedx_path="/nonexistent"))
    state=State{Float64}(energy=prevfloat(100.),charge=1.,distance=1000.)
    c=prepare_mixed_kernel(p,1)
    # Force the positive component of the small-step fluctuation mixture,
    # then place hard-event optical depths beyond both requested pieces.
    boundary_rng=MixedScriptedRNG([1e-12,.5,1e10,1e10])
    final=DiffPumas.Tomography.propagate_mc_uniform_segment(p,state,1,2650.,.001,boundary_rng;
        straggling=true,energy_threshold_low=100.,kernel_context=c)
    @test final.energy>=100.
    @test final.distance-state.distance ≈ .001 atol=4eps(state.distance)
    @test final.grammage ≈ 2.65 rtol=1e-10
    killed=MixedKernel(p.mass,p.ctau,p.settings.cutoff,[1.,1e6],[2.,2.],[.1,.1],
        [0.,(1e6-1)/2],[0.,(1e6-1)*.05],[0.,0.],(),1.,WeakRef(p.tables),((1,),(1.,)))
    initial=State{Float64}(energy=1000.,charge=1.,distance=1000.)
    result=DiffPumas.Tomography.propagate_mc_uniform_segment(p,initial,1,2650.,.001,
        MixedScriptedRNG([0.,.5]);straggling=true,energy_threshold_low=100.,kernel_context=killed)
    @test result.weight==0
    @test result.distance==initial.distance
end

@testset "Adaptive segment floor never overshoots a short remainder" begin
    for remaining in (2e-8,5e-7,1e-6,1e-4), maximum_step in (1e-10,remaining/2,1.)
        step=DiffPumas.Tomography._mc_segment_step(remaining,maximum_step)
        @test 0<step<=remaining
    end
end
