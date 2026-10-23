using DiffPumas,Test,SHA
using DiffPumas.Physics: property_stopping_power,property_straggling,property_cross_section
const T=DiffPumas.Tomography
include_string(T,read(joinpath(@__DIR__,"candidate.jl"),String),"external_shallow_candidate.jl")
candidate=T.mc_cell_mixture_and_density_candidate
original=T.mc_cell_mixture_and_density
cfg=T.MaterialConfig(1,2,3,3,2700.,1000.,1500.,50.)
p=create_physics(MUON;n_energies=4,K_min=1.,K_max=100.,materials=[STANDARD_ROCK,WATER,AIR],settings=PhysicsSettings(dedx_path="/nonexistent"))
# Deliberately distinct synthetic constant properties, not empirical accuracy data.
csda=[.002,.004,.008];mixed=[.001,.003,.005];variance=[.01,.03,.09];hard=[.02,.05,.11]
for (i,t) in enumerate(p.tables)
 t.csda_stopping_power .= csda[i];t.mixed_stopping_power .= mixed[i]
 t.straggling .= variance[i];t.cross_section .= hard[i]
end
@testset "Shallow candidate endpoint and material properties" begin
 for w in (0.,5e-13,1e-12,2e-12,1e-8,.3,1.)
  mix,rho=candidate(w,true,cfg)
  expected=[.5*(1-w),w,.5*(1-w)] # order rock, water, porous
  @test rho≈1/sum(expected./[2700.,1000.,1500.]) rtol=2e-15
  @test rho≈T.cell_density(w,true,cfg) rtol=2e-15
  for (property,values) in ((m->property_stopping_power(p,ENERGY_LOSS_CSDA,m,10.),csda),(m->property_stopping_power(p,ENERGY_LOSS_MIXED,m,10.),mixed),(m->property_straggling(p,m,10.),variance),(m->property_cross_section(p,m,10.),hard))
   @test property(mix)≈sum(expected.*values) rtol=2e-15
  end
  @test property_stopping_power(p,ENERGY_LOSS_CSDA,mix,10.)≈T.cell_stopping_power(p,w,true,cfg,10.) rtol=2e-15
 end
 # Original must fail the same declared endpoint physics; positive branch unchanged.
 old,oldrho=original(0.,true,cfg);dry,dryrho=candidate(0.,true,cfg)
 @test oldrho!=dryrho
 @test property_stopping_power(p,ENERGY_LOSS_CSDA,old,10.)!=property_stopping_power(p,ENERGY_LOSS_CSDA,dry,10.)
 for w in (2e-12,.3,1.)
  a,ar=original(w,true,cfg);b,br=candidate(w,true,cfg)
  @test a.materials==b.materials && a.fractions==b.fractions && ar==br
 end
 for disabled in (T.MaterialConfig(1,2,3,3,2700.,1000.,1500.,0.),T.MaterialConfig(1,2,3,0,2700.,1000.,1500.,50.),cfg), w in (0.,5e-13,.3)
  shallow=disabled!==cfg
  a,ar=original(w,shallow,disabled);b,br=candidate(w,shallow,disabled)
  @test a.materials==b.materials && a.fractions==b.fractions && ar==br
 end
end
@testset "Constant stopping transport and endpoint derivative" begin
 distance=.2;energy0=10.
 function energy(w)
  mix,rho=candidate(w,true,cfg)
  return energy0+distance*rho*property_stopping_power(p,ENERGY_LOSS_CSDA,mix,energy0)
 end
 # Analytic homogeneous slab result and derivative from specific-volume addition.
 a=.5/2700+.5/1500;b=1/1000-a;s0=.5*(csda[1]+csda[3]);ds=csda[2]-s0
 for w in (0.,1e-12,.3)
  e,_,ok=T.propagate_csda_cell_segment(p,w,true,cfg,distance,energy0,1.)
  @test ok
  @test e≈energy(w) rtol=1e-14
  exact=distance*(ds*a-s0*b)/(a+b*w)^2
  h=1e-5
  numerical=(-3energy(w)+4energy(w+h)-energy(w+2h))/(2h)
  @test numerical≈exact rtol=2e-8
 end
 @test abs(energy(2e-12)-energy(0.))<1e-10
end
println("Candidate evaluated under separate name; tracked method untouched. Deterministic consistency only, not stochastic validation.")
