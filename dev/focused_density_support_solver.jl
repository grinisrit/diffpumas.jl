# Isolated production-equivalent active-set GN, with trial logging/deadline only.
# Production formula: src/Tomography_inverse.jl::gauss_newton_reconstruct.
function focused_bounded_gn(initial,make_state,predict_trial,prior,support;
        max_outer=10,deadline=Inf,trace_callback=identity,state_callback=(args...)->nothing,
        tolerance=1e-6,clock=time)
    T=DiffPumas.Tomography;active=findall(support);state=initial;mu=1e-3
    accepted_count=0;outer_count=0;reason="outer_budget_complete"
    for outer in 1:max_outer
        clock()>=deadline && (reason="soft_deadline_before_outer";break)
        outer_count=outer
        Ja=state.J[:,active];normal=Matrix(transpose(Ja)*Ja)
        gradient=transpose(Ja)*state.residual+laplacian_gradient(prior,state.water)[active]
        normal.+=Matrix(T._prior_hessian(prior,state.water)[active,active])
        pg=state.water[active].-clamp.(state.water[active].-gradient,0.,MAX_WATER_FRACTION)
        if norm(pg,Inf)<=tolerance;reason="selected_branch_tolerance_met";break;end
        movable=.!(((state.water[active].<=0).&(gradient.>0)).|((state.water[active].>=MAX_WATER_FRACTION).&(gradient.<0)))
        diagonal=diag(normal);dmax=maximum(diagonal);dmax=dmax>0 ? dmax : 1.
        accepted=false;budget_stop=false
        for lm_try in 1:7
            rescue=lm_try==7;fallback=false;fallback_reason="none"
            A=normal+Diagonal(mu.*max.(diagonal,1e-12*dmax))
            delta=rescue ? -gradient./max.(diagonal,maximum(abs.(gradient)),1e-30) : try
                step=zeros(length(active));step[movable].=-(A[movable,movable]\gradient[movable])
                projected=clamp.(state.water[active].+step,0.,MAX_WATER_FRACTION).-state.water[active]
                if dot(gradient,projected)>=0;step.=-gradient./max.(diagonal,1e-12*dmax);fallback=true;fallback_reason="projected_newton_not_descent";end
                step
            catch err
                fallback=true;fallback_reason="linear_solve_"*string(typeof(err));-gradient./max.(diagonal,1e-12*dmax)
            end
            if !all(isfinite,delta)
                trace_callback((outer=outer,lm_try=lm_try,halving=0,alpha=0.,mu=mu,rescue=rescue,fallback=fallback,fallback_reason=fallback_reason,
                    objective_before=state.objective,trial_objective=NaN,confirmed_objective=NaN,data=NaN,prior=NaN,
                    normalization=NaN,directional=NaN,step_inf=NaN,accepted=false,reason="nonfinite_direction"))
                mu=min(mu*3.,1e12);continue
            end
            alpha=1.;cached_water=nothing;cached_trial=nothing
            for halving in 1:(rescue ? 40 : 14)
                if clock()>=deadline;reason="soft_deadline_before_trial";budget_stop=true;break;end
                candidate=copy(state.water);candidate[active].=clamp.(state.water[active].+alpha.*delta,0.,MAX_WATER_FRACTION)
                displacement=candidate[active].-state.water[active];directional=dot(gradient,displacement)
                base=(outer=outer,lm_try=lm_try,halving=halving,alpha=alpha,mu=mu,rescue=rescue,fallback=fallback,fallback_reason=fallback_reason,
                    objective_before=state.objective,directional=directional,step_inf=norm(displacement,Inf))
                if candidate==state.water
                    trace_callback(merge(base,(trial_objective=state.objective,confirmed_objective=NaN,data=state.data,
                        prior=state.prior,normalization=state.normalization,accepted=false,reason="zero_projected_step")))
                    break
                end
                trial= cached_water!==nothing && candidate==cached_water ? cached_trial : predict_trial(candidate)
                cached_water=candidate;cached_trial=trial
                confirmed=NaN;take=false
                rejection=!isfinite(trial.objective) ? "nonfinite_objective" : directional>=0 ? "not_branch_descent" :
                    trial.objective>=state.objective ? "not_strict_decrease" : trial.objective>state.objective+1e-4*directional ? "armijo_failure" : "candidate_pass"
                if clock()>=deadline
                    reason="soft_deadline_after_prediction";budget_stop=true;rejection=reason
                elseif T._gn_accept_trial(trial.objective,state.objective,gradient,displacement,:active_set)
                    full=make_state(candidate);confirmed=full.objective
                    if clock()>=deadline
                        reason="soft_deadline_during_confirmation";budget_stop=true;rejection=reason
                    elseif T._gn_accept_trial(full.objective,state.objective,gradient,displacement,:active_set)
                        take=true;rejection="accepted_full_confirmation"
                        state=full;accepted=true;accepted_count+=1
                        state_callback(state,accepted_count,outer)
                    else
                        rejection="full_confirmation_rejected"
                    end
                end
                trace_callback(merge(base,(trial_objective=trial.objective,confirmed_objective=confirmed,data=trial.data,
                    prior=trial.prior,normalization=trial.normalization,accepted=take,reason=rejection)))
                (accepted || budget_stop) && break
                alpha*=.5
            end
            if accepted;mu=max(mu/3.,1e-12);break
            elseif budget_stop;break
            else;mu=min(mu*3.,1e12);end
        end
        budget_stop && break
        if !accepted;reason="all_directions_rejected";break;end
    end
    (state=state,accepted=accepted_count,outer_attempts=outer_count,termination=reason)
end
