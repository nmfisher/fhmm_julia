# Structured variational inference
const ϵ = 1.0e-64


# update variational parameter `h` given expectation `γ`
function _update_svp!(fhmm::FHMM,
                      Y::AbstractMatrix,
                      h::Array{Float64, 3}, # shape (K, T, M)
                      γ::Array{Float64, 3}, # shape (K, T, M) 
                      C⁻¹::Matrix{Float64},  # shape (D, D)
                      Δ::Matrix{Float64}, # shape (K, M)
                      Ỹ::Array{Float64, 3} # shape (K, M, T)
                      )
    D, T = size(Y)
    # alias
    W = fhmm.W
    
    for m=1:fhmm.M
        Δ[:,m] = diag(W[:,:,m]'*C⁻¹*W[:,:,m])
    end
    
    #display(Δ)

    # Eq. (12b)
    ΣWγ = zeros(D)
    for m=1:fhmm.M
        for t=1:T
            fill!(ΣWγ, 0.0)
            for l=1:fhmm.M
                if m != l
                    ΣWγ += W[:,:,l] * γ[:,t,l]
                end
            end
            Ỹ[:,m,t] = Y[:,t] - ΣWγ
        end
    end
    
    #display("Ỹ")
    #display(Ỹ)
    #display("W")
    #display(W)
    

    @assert !any(isnan.(Ỹ))
    display("h-old")
    display(h)

    # Eq. (12a)
    for m=1:fhmm.M
        WᵗC⁻¹ = W[:,:,m]'C⁻¹ # M x D
        #display("WᵗC⁻¹")
        #display(WᵗC⁻¹)
        for t=1:T
            h[:,t,m] = WᵗC⁻¹*Ỹ[:,m,t] - 0.5Δ[:,m]
            #h[:,t,m] .-= maximum(h[:,t,m])
            #h[:,t,m] = softmax(h[:,t,m])
            #h[:,t,m] ./= sum(h[:,t,m])
        end
    end
    #display("h-new pre exp")
    #display(h)
    h = exp.(h)
    #display("h-new post exp")
    #h = softmax(h,dims=1)
    #display(h)
        
    @assert !any(isnan.(h))
    @assert !any(isinf.(h))
    #@assert(all(isapprox.(sum(h,dims=1), 1)))
    return h
end

# update expectations using forward-backward algorithm
function _updateE!(fhmm::FHMM,
                   m::Int,
                   Y::AbstractMatrix,    # shape: (D, T)
                   α::Array{Float64, 3},   # shape: (K, T, M)
                   β::Array{Float64, 3},   # shape: (K, T, M)
                   γ::Array{Float64, 3},   # shape: (K, T, M)
                   ξ::Array{Float64, 4}, # shape: (K, K, T-1, M)
                   h::Array{Float64, 3})   # shape: (K, T, M)
    return
    D, T = size(Y)
    K = fhmm.K
    display("α")
    display(α)
    display("h")
    display(h)
    display("β")
    display(β)
    display("γ")
    display(γ)
    display("ξ")
    display(ξ)
    # scaling paramter
    c = zeros(Float64, T)
    # forward recursion
    α[:,1,m] = fhmm.π[:,m] .* h[:,1,m] / (c[1] = sum(α[:,1,m]))
    for t=2:T
        @inbounds α[:,t,m] = (fhmm.P[:,:,m]' * α[:,t-1,m]) .* h[:,t,m]
        @inbounds α[:,t,m] /= (c[t] = sum(α[:,t,m]))
    end
    
    @assert !any(isnan.(α[:,:,m]))
    
    # backword recursion
    β[:,T,m] .= 1.0
    for t=T-1:-1:1
        β[:,t,m] = fhmm.P[:,:,m] * β[:,t+1,m] .* h[:,t+1,m] ./ c[t+1]
    end
    @assert !any(isnan.(β[:,:,m]))

    γ[:,:,m] = α[:,:,m] .* β[:,:,m]

    for t=1:T-1
        ξ[:,:,t,m] = fhmm.P[:,:,m] .* α[:,t,m] .* β[:,t+1,m]' .* 
            h[:,t+1,m]' ./ c[t+1]
    end

    @assert(all(isapprox.(sum(γ, dims=(1)), 1, atol=0.0001)))
    display("e step done")
end

# update variational paramter until convergence and then update expectations.
function updateE!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  α::Array{Float64, 3}, # inplace
                  β::Array{Float64, 3}, # 
                  γ::Array{Float64, 3}, # shape (K, T, M)
                  ξ::Array{Float64, 4}, # inplace
                  h::Array{Float64, 3};  # shape (K, T, M)
                  maxiter::Int=100,
                  tol::Float64=1.0e-4,
                  verbose::Bool=false)
    D, T = size(Y)
    K = fhmm.K
    M = fhmm.M

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    
    Δ = zeros(K, M)
    Ỹ = zeros(D, M, T)

    hᵗ = copy(h)
    for iter=1:maxiter
        # update variational parameters
        hᵗ = _update_svp!(fhmm, Y, hᵗ, γ, C⁻¹, Δ, Ỹ)
        
        # check if converged
        diff = norm(hᵗ-h)
        display("$(diff): converged at #$(iter).")
        if diff < tol
            break
        end

        # update h with new one
        h = copy(hᵗ)
    end
    
    # update expectations using forward-backward algorithm for 
    # each factorized HMM
    for m=1:M
        _updateE!(fhmm, m, Y, α, β, γ, ξ, h)
    end
    
    return γ, ξ, h
end

# M-step
function updateM!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  γ::Array{Float64, 3}, # shape (K, T, M)
                  ξ::Array{Float64, 4}; # shape (K, K, T-1, M)
                  fzero::Bool=false)
    D, T = size(Y)
    M = fhmm.M
    K = fhmm.K
    # Eq. (A.4) update initial state prob.
    for m=1:M
        fhmm.π[:,m] = γ[:,1,m] / sum(γ[:,1,m] .+ ϵ)
    end
    
    #display("fhmm.π")
    #display(fhmm.π)
    
    #display("γ")
    #display(γ)
    
    # Eq. (A.5) update transition matrices
    for m=1:fhmm.M
        fhmm.P[:,:,m] = sum(ξ[:,:,:,m], dims=3) ./ (sum(γ[:,1:end-1,m], dims=2) .+ ϵ)
        fhmm.P[:,:,m] ./= sum(fhmm.P[:,:,m], dims=1)
    end
    
    #display("fhmm.P")
    #display(fhmm.P)

    # Eq. (A.3) update observation means
    #tmp = zeros(fhmm.D, K*M, T)
    γʳ = zeros(K*M, T)
    for t=1:T
       γʳ[:,t] = reshape(γ[:,t,:], K*M)
       #tmp[:,:,t] = Y[:,t] * γʳ[:,t]'
    end
    #display(Y*γʳ' == reshape(sum(tmp, dims=3)
    #display((Y*γʳ') * pinv(γʳ*γʳ'))
    display("fhmm.W pre-mstep")
    display(fhmm.W)
    fhmm.W[:,:,:] = reshape((Y*γʳ') * pinv(γʳ*γʳ'), D, K, M)
    display("fhmm.W post-mstep")
    display(fhmm.W)
    # Eq. (A,7) update C
    s = zeros(D, D)
    for t=1:T
        for m=1:fhmm.M
            s += fhmm.W[:,:,m]*γ[:,t,m]*Y[:,t]'
        end
    end    
    #display("s")
    #display(s)
    #for d=1:fhmm.D
    fhmm.C[:,:] = 1/T * (Y*Y' - s)
    #end
    display("fhmm.C post m-step")
    display(fhmm.C)
    #@assert !any(fhmm.C .< 1e-11 )
    nothing
end

# TODO derive correct lower bound
function bound_sv(fhmm::FHMM,
                  Y::AbstractMatrix,
                  γ::Array{Float64, 3},
                  ξ::Array{Float64, 4})
    D, T = size(Y)
    likelihood::Float64 = 0.0
    logγ::Array{Float64, 3} = log.(γ .+ ϵ)
    logP::Array{Float64, 3} = log.(fhmm.P .+ ϵ)
    
    for t=1:T
        for m=1:fhmm.M
            likelihood += (γ[:,t,m]'logγ[:,t,m])[1]
        end
    end

    likelihood -= T*D*log(pi)
    d = det(fhmm.C)
    if d > 0
        likelihood -= 0.5T*log(d)
    end

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    for t=1:T
        ws = sum([fhmm.W[:,:,m]*γ[:,t,m] for m=1:fhmm.M])
        likelihood -= (0.5*(Y[:,t] - ws)'*C⁻¹*(Y[:,t] - ws))[1]
    end

    for m=1:fhmm.M
        likelihood += (γ[:,1,m]'log.(fhmm.π[:,m] .+ ϵ))[1]
    end

    #=
    for t=2:T
        for m=1:fhmm.M
            likelihood += (θ[:,m,t]'logP[:,:,m]*θ[:,m,t-1])[1]
        end
    end
    =#

    @assert !isnan(likelihood)

    return likelihood
end

# Result of structured variational (SV) inference
struct SVInferenceResult
    likelihoods::Vector{Float64}
    h::Array{Float64, 3}
    α::Array{Float64, 3}
    β::Array{Float64, 3}
    γ::Array{Float64, 3}
    ξ::Array{Float64, 4}
end

function fit_sv!(fhmm::FHMM,
                 Y::AbstractMatrix; # observation matrix, shape (D, T)
                 maxiter::Int=100,
                 tol::Float64=1.0e-5,
                 fzero::Bool=false,
                 verbose::Bool=false)
    D, T = size(Y)
    M = fhmm.M
    K = fhmm.K
        
    # means of observation probability
    α = rand(K, T, M)
    α ./= sum(α, dims=1)
    β = rand(K, T, M)
    β ./= sum(β, dims=1)
    γ = rand(K, T, M) 
    γ ./= sum(γ,dims=(1))
    
    ξ = rand(K, K, T-1, M) 
    ξ ./= sum(ξ, dims=(1,2))
    h = rand(K, T, M) 
    h ./= sum(h,dims=(1))
    #display(ξ)
    #display(sum(ξ, dims=(1,2)))
    #@assert(all(sum(ξ, dims=(1,2)) .== 1))
    #@assert(all(sum(γ, dims=(1)) .== 1))

    likelihood::Vector{Float64} = zeros(1)
    score = 1
    # Roop of EM algorithm
    for iter=1:maxiter
        display("Iter $iter")
        # compute bound
        #score = bound_sv(fhmm, Y, γ, ξ)
        
        # update expectations
        #γ, ξ, h = 
        updateE!(fhmm, Y, α, β, γ, ξ, h, verbose=verbose)
        
        # update parameters of FHMM
        updateM!(fhmm, Y, γ, ξ, fzero=fzero)

        improvement = (score - likelihood[end]) / abs(likelihood[end])

        if verbose
            println("#$(iter): bound $(likelihood[end])
                    improvement: $(improvement)")
        end

        # check if converged
        if iter > 1 && improvement < 1.0e-7
            if verbose
                println("#$(iter) converged")
            end
            break
        end

        push!(likelihood, score)
    end

    # remove initial zero
    popat!(likelihood, 1)

    return SVInferenceResult(likelihood, h, α, β, γ, ξ)
end

