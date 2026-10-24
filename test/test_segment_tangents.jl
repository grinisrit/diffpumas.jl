using Test, DiffPumas, DiffPumas.Tomography, ForwardDiff, ChainRulesCore

@testset "Piecewise CSDA segment tangents" begin
    T=DiffPumas.Tomography; P=DiffPumas.Physics
    original_mode=get_adjoint_derivative_mode(); original_step=get_adjoint_fd_step()
    try
        @test original_mode == :forwarddiff
        @test_throws ArgumentError set_adjoint_derivative_mode!(:unknown)
        set_adjoint_fd_step!(1e-5)
        @test get_adjoint_derivative_mode() == :forwarddiff
        # Equal primal values must not let tangent signs choose a branch.
        for (a,b) in ((1.,2.),(-2.,1.))
            left=ForwardDiff.Dual(1.,a);right=ForwardDiff.Dual(1.,b)
            @test ForwardDiff.partials(T._csda_min(left,right))[1] == a
            @test ForwardDiff.partials(T._csda_max(left,right))[1] == a
            @test T._csda_min(1.,left) == 1.
        end
        @test isequal(T._csda_min(0.,-0.),min(0.,-0.))
        @test isequal(T._csda_max(0.,-0.),max(0.,-0.))
        xs=[1.,2.,4.]; ys=[1.,3.,2.]
        at(x)=P.interpolate_table_fast(x,log(x),xs,log.(xs),ys)
        @test ForwardDiff.derivative(at,1.) == 0.
        @test ForwardDiff.derivative(at,4.) == 0.
        # Interior interpolation knots take the primal-selected right interval.
        @test ForwardDiff.derivative(at,2.) ≈ -1/(2log(2.))

        ph=create_physics(MUON;n_energies=80,K_min=.001,K_max=1e6,
            materials=[STANDARD_ROCK,AIR,WATER])
        rock=get_material_index(ph,"StandardRock"); water=get_material_index(ph,"Water");air=get_material_index(ph,"Air")
        mat=MaterialConfig(rock,water,air,-1,Float64(ph.tables[rock].density),Float64(ph.tables[water].density),0.,0.)
        site=SiteConfig(963.,30000.)
        e=123.4567;wf=.21347; h=1e-5
        cell=(w,en,weight)->T.propagate_csda_cell_segment(ph,w,false,mat,33.3,en,weight)
        uniform=(en,weight)->T.propagate_csda_uniform_segment(ph,rock,mat.rock_density,333.,en,weight)
        airfun=(en,weight)->T.propagate_csda_air_segment(ph,air,site,1000.,.8,10000.,en,weight)
        for mode in (:forwarddiff,:finite_difference)
            set_adjoint_derivative_mode!(mode)
            set_adjoint_fd_step!(1e-7)
            out,pb=ChainRulesCore.rrule(T.propagate_csda_cell_segment,ph,wf,false,mat,33.3,e,0.)
            cot=pb((1.,1.,NoTangent()))
            # Zero incoming weight kills composition dependence of weight, but
            # neither the energy map nor derivative with respect to input weight.
            efd=(cell(wf,e+h,0.)[1]-cell(wf,e-h,0.)[1])/(2h)
            wfd=(cell(wf+h,e,0.)[1]-cell(wf-h,e,0.)[1])/(2h)
            @test cot[7] ≈ efd rtol=1e-6
            @test cot[3] ≈ wfd rtol=1e-6
            @test cot[8] ≈ cell(wf,e,1.)[2] rtol=1e-14
            @test out[2] == 0.
            for (func,args,energy_index,weight_index,primal) in (
                (T.propagate_csda_uniform_segment,(ph,rock,mat.rock_density,333.,e,0.),6,7,uniform),
                (T.propagate_csda_air_segment,(ph,air,site,1000.,.8,10000.,e,0.),8,9,airfun))
                _,pull=ChainRulesCore.rrule(func,args...)
                v=pull((1.,1.,NoTangent()))
                @test v[energy_index] ≈ (primal(e+h,0.)[1]-primal(e-h,0.)[1])/(2h) rtol=1e-6
                @test v[weight_index] ≈ primal(e,1.)[2] rtol=1e-14
            end
        end
        set_adjoint_derivative_mode!(:forwarddiff)
        for distance in (0.,33.3), w in (0.,wf,.8)
            out= T.propagate_csda_cell_segment(ph,w,false,mat,distance,e,1.)
            dual=T.propagate_csda_cell_segment(ph,ForwardDiff.Dual(w,1.,0.),false,mat,distance,ForwardDiff.Dual(e,0.,1.),ForwardDiff.Dual(1.,0.,0.))
            @test ForwardDiff.value(dual[1]) == out[1]
            @test ForwardDiff.value(dual[2]) == out[2]
            _,pb=ChainRulesCore.rrule(T.propagate_csda_cell_segment,ph,w,false,mat,distance,e,0.)
            if distance==0.
                cot=pb((1.,1.,NoTangent()))
                @test cot[3] == 0.
                @test cot[7] == 1.
                @test cot[8] == 1.
            end
        end
    finally
        set_adjoint_fd_step!(original_step)
        set_adjoint_derivative_mode!(original_mode)
    end
end
