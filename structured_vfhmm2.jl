# estimate P(Mt|Yt,θ) to calculate <Mt>
function forward_backward(
    fhmm::FHMM, 
    Y:AbstractMatrix, # (DxT),
    h::Array{Float64,3}, # (KxTxM) 
    α::Array{Float64,3} #(KxTxM)
    β::Array{Float64,3} # (KxTxM)
    γ::Array{Float64,3} # (KxTxM) 
    ξ::Array{Float64, 4} # (KxKxT-1xM)
)
    T=size(Y)[2]
    
    <S> = zeros(fhmm.K, fhmm.M,T)
    # P(Mt1) is h
    # therefore we just need to calculate F/B 
    # P(Mt1=K1|Yt1) = ( P(Yt|Mt1=K1) * P(Yt1) ) / P(Mt1=K1)
    # P(Mt1=K2|Yt1) = ( P(Yt|Mt1) * P(Yt1) ) / P(Mt1=K2)
    # P(Yt|Mt1=K1) = 
    for t=1:T
        while(true)
            if(m == M)
                if(size(stack)[1] != M - 1)
                    break
                end
                for s=1:K
                    push!(stack, s)
                    mu = zeros(D)
                    for i=1:M
                        mu += model.W[:, stack[i], i]
                    end
                    
                    p_yt_given_mt_in_s = pdf(MvNormal(mu, Array{Float64,2}(Hermitian(Symmetric(model.C)))), X[:,t])
                    p_mt_in_s = reduce(*, θ[stack,:,t])
                    unnormalized = p_yt_given_mt_in_s * p_mt_in_s
                    if unnormalized > pmax
                        pmax = unnormalized
                        mmax = copy(stack)
                    end
                    pop!(stack)
                end
                stack[end] +=1
                while(stack[end] > K)
                    pop!(stack)
                    if(size(stack)[1] == 0)
                        break
                    end
                    stack[end] +=1
                    m -= 1
                end
            else
                push!(stack, 1)
                m += 1
            end
        end
        states[:,t] = mmax
    end
    return states
    #forward 

function update_variational_parameters(
    fhmm::FHMM, 
    h::Array{Float64,3}, # (KxTxM) 
    Y::AbstractMatrix, # (DxT)
    γ::Array{Float64, 3}) # (KxTxM)
    
    C⁻¹::Matrix{Float64} = fhmm.C^-1
    T=size(Y)[2]
    Δ = zeros(fhmm.K, 1, fhmm.M)
    
    for m=1:fhmm.M
        #            KxD                     DxD        D,K
        res = transpose(fhmm.W[:,:,m]) * C⁻¹ * fhmm.W[:,:,m]
        Δ[:,1, m] = diag(res)
    end
    
    Ỹ = zeros(fhmm.D,fhmm.M,T)
    obs = zeros(fhmm.D,fhmm.M,T)
    for t=1:size(Y)[2]
        for m=1:fhmm.M    
            for l=1:fhmm.M
                if l != m
                    obs[:,m,t] += fhmm.W[:,:,m] * h[:,t,m] # (DxK)*(Kx1) <-- IS THIS CORRECT? Is h the expectation of S over l?
                end
            end
        end
    end

    Ỹ = Y - reshape(sum(obs, dims=2), fhmm.D, fhmm.D)  # (DxT)
    h_new = copy(h)
    for t=1:size(Y)[2]
        for m=1:fhmm.M
           # Kx1x1         (KxD)              * (DxD) * (Dx1)
           h_new[:,t,m] = transpose(fhmm.W[:,:,m]) * C⁻¹ * Ỹ[:,t] # (KxD) x (DxD) x (DxD)
        end
    end
    #display(h_new)
    #display(Δ)
    # KxTxM = KxTxM - KxM
    h_new = exp.(h_new) .- 0.5Δ
    delta = sum(h_new - h)
    display("Delta: $delta")
    return h_new
end


#function compute_likelihood(fhmm::FHMM, Y::AbstractMatrix; # observation matrix, shape (D, T))
#    return
#end

function fit_sv2!(fhmm::FHMM,
                 Y::AbstractMatrix; # observation matrix, shape (D, T)
                 maxiter::Int=100,
                 tol::Float64=1.0e-5,
                 fzero::Bool=false,
                 verbose::Bool=false)
    D, T = size(Y)
    M = fhmm.M
    K = fhmm.K
        
    # holders for forward-backward step 
    α = zeros(K, T, M)
    β = zeros(K, T, M)
    γ = zeros(K, T, M) 
    ξ = zeros(K, K, T-1, M) 
    
    # initial values for variational parameter h (probability of a fictitious observation for every chain being in every state at every timestep)
    h = rand(K, T, M) 
    h ./= sum(h,dims=(1))
    @assert(all(sum(h, dims=(1)) .== 1))
    
    # holder for likelihood so we can compare improvement across 
    likelihood::Vector{Float64} = zeros(1)
    for i=1:maxiter
        likelihood_i = compute_likelihood()
    end
end
