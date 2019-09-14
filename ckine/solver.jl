using gcSolver

@Base.ccallable function runCkineC(tpsIn::Ptr{Cdouble}, nTps::Csize_t, outt::Ptr{Cdouble}, params::Ptr{Cdouble}, paramsLen::Csize_t)::Cint
    vecc = unsafe_wrap(Vector{Float64}, params, paramsLen)
    tps = unsafe_wrap(Vector{Float64}, tpsIn, nTps)
    
    output = runCkine(tps, vecc)
    
    outWrap = unsafe_wrap(Array{Cdouble, 2}, outt, size(output))
    
    outWrap[:, :] .= output[:, :]
    
    return 0
end
