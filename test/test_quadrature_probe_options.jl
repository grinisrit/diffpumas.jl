using Test
include(joinpath(@__DIR__,"..","dev","energy_quadrature_probe.jl"))
@testset "Quadrature probe explicit immutable inputs" begin
    mktempdir() do directory
        dump=joinpath(directory,"physics.dump");write(dump,"input fixture")
        calibration=joinpath(directory,"calibration.toml");write(calibration,"enabled = false\n")
        output=joinpath(directory,"new")
        base=["--dump",dump,"--calibration",calibration]
        options,zeniths=quadrature_probe_options(vcat(base,["--output",output,"--zeniths","0,30,45,60,75"]))
        @test options["dump"]==dump && options["calibration"]==calibration
        @test zeniths==[0,30,45,60,75]
        @test !ispath(output)
        _,historical_angles=quadrature_probe_options(vcat(base,[output]))
        @test historical_angles==[0,30,45]
        @test_throws ErrorException quadrature_probe_options(base)
        @test_throws ErrorException quadrature_probe_options(vcat(base,["--output",directory]))
        @test_throws ErrorException quadrature_probe_options(vcat(base,[output,"--dump",joinpath(directory,"missing")]))
        for invalid in ("0,90","0,NaN","0,0")
            @test_throws ErrorException quadrature_probe_options(vcat(base,[output,"--zeniths",invalid]))
        end
    end
    emap=LVDTopo.build_elevation_map();mat=MaterialConfig(1,2,3,4,2710.0,1000.0,0.0,0.0)
    _,paths,coverage=quadrature_probe_geometry(emap,mat,[0.0,30.0,45.0,60.0,75.0])
    @test length(coverage)==5
    @test all(p->p.valid && !isempty(p.segments),paths)
    @test all(row->row.slant_m>0 && row.in_volume_m>0 && row.remaining_rock_m>=0,coverage)
end
