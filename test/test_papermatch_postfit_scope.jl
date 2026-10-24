# Standalone: julia --project=. test/test_papermatch_postfit_scope.jl
using Test, DiffPumas, LinearAlgebra

@testset "Stable paper-match operator survives recursion scope" begin
    source = read(joinpath(@__DIR__, "..", "examples", "lvd_tomography.jl"), String)
    # Execute the actual declaration, factory assignment and post-loop call in
    # their original function/for/if scopes. Omit expensive geometry/optimization
    # and report I/O; the synthetic forward model makes no transport claim.
    declaration = only(collect(eachmatch(r"recursion_converged = false\n(.*?)    for recur in 1:n_recur"s, source))).captures[1]
    assignment = only(collect(eachmatch(r"            robust_model, profile_scale, robust_predict = .*?return_predict_only=true\)"s, source))).match
    postcall = only(collect(eachmatch(r"        _,stable_J=robust_model\(w_matched\)", source))).match
    function scope_expression(name, outer_declaration)
        Meta.parse("""
        function $name(stable_fit, n_recur, model_aq, meas_v, fit_mask, wdem, args, predict_aq, w_matched)
            $outer_declaration
            for recur in 1:n_recur
                if stable_fit
                    $assignment
                end
            end
            if stable_fit
                $postcall
                return transpose(stable_J)*stable_J
            end
            return nothing
        end
        """)
    end
    Core.eval(@__MODULE__, scope_expression(:papermatch_scope_fixed, declaration))
    Core.eval(@__MODULE__, scope_expression(:papermatch_scope_old, ""))
    A = [1.0 0.2; -0.3 1.0; 0.6 -0.7]
    model = w -> (exp.(A*w), exp.(A*w).*A)
    observations = 2.5 .* exp.(A*[0.3, 0.6])
    weights = [1.0, 0.5, 0.8]
    args = (papermatch_log_scale=0.3, papermatch_abs_scale=0.0)
    predict = w -> first(model(w))
    w = [0.1, 0.4]
    operator, _ = DiffPumas.Tomography.profiled_log_observation_operator(model, observations;
        weights=weights, robust_scale=args.papermatch_log_scale, normalization=args.papermatch_abs_scale)
    _, expected_J = operator(w)
    inputs = (model, observations, ones(3), weights, args, predict, w)
    @test papermatch_scope_fixed(true, 1, inputs...) ≈ expected_J' * expected_J
    @test papermatch_scope_fixed(true, 2, inputs...) ≈ expected_J' * expected_J
    @test papermatch_scope_fixed(false, 1, inputs...) === nothing
    @test_throws UndefVarError papermatch_scope_old(true, 1, inputs...)
end
