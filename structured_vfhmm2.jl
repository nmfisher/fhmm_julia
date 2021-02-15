# we need to calculate P(Mt|Yt,θ) for E_Mt
function forward_backward(
    fhmm::FHMM, 
    Y::AbstractMatrix, # (DxT),
    h::Array{Float64,3}) # (KxTxM) )
    
    T=size(Y)[2]
    γ = zeros(fhmm.M,fhmm.K,T) 
    α = zeros(fhmm.K, T, fhmm.M)
    β = zeros(fhmm.K, T, fhmm.M)
    ξ = zeros(fhmm.M, fhmm.K, fhmm.K, T-1)
    exp_s = zeros(fhmm.M, fhmm.K, fhmm.K, T-1)
    
    # forward recursion
    for m=1:fhmm.M
      for k=1:fhmm.K
        α[k,1,m] = h[k,1,m] * fhmm.π[k,m]
      end
    end
    
    for t=2:T
      for m=1:fhmm.M
        for k=1:fhmm.K
          for j=1:fhmm.K
            α[k,t,m] = h[k,t,m] * α[j,t-1,m]
          end
        end
      end
    end
       
    β[:,T,:] .= 1
    C⁻¹ = fhmm.C^1
    # backwards
    for t=T-1:-1:1
      for m = 1:fhmm.M
        for k = 1:fhmm.K
          for j = 1:fhmm.K
            β[k,t,m] += fhmm.P[k,j,m] * h[k,t,m] * β[j,t+1,m]
          end
        end
      end
    end
    

    
    for t=1:T
      for m=1:fhmm.M
        for k=1:fhmm.K
          γ[m,k,t] = α[k,t,m] * β[k,t,m]
          if t < T
            for j=1:fhmm.K
              ξ[m,k,j,t] = α[k,t,m] * fhmm.P[k,j,m] * β[j,t+1,m] * h[j,t+1,m]
            end
          end
        end
        γ[m,:,t] ./= sum(γ[m,:,t])
        if t < T
          ξ[m,:,:,t] ./= sum(ξ[m,:,:,t])
        end
      end
    end
    return γ, ξ
end

function update_variational_parameters(
    fhmm::FHMM, 
    h::Array{Float64,3}, # (KxTxM) 
    Y::AbstractMatrix, # (DxT)
    γ::Array{Float64, 3}) # (MxKxT)
    fhmm.C[:,:] .+= 1.0e-64
    C⁻¹::Matrix{Float64} = fhmm.C^-1
    T=size(Y)[2]
    Δ = zeros(fhmm.K, 1, fhmm.M)
    
    for m=1:fhmm.M
        res = transpose(fhmm.W[:,:,m]) * C⁻¹ * fhmm.W[:,:,m]
        Δ[:,1, m] = diag(res)
    end
    
    Ỹ = zeros(fhmm.D,fhmm.M,T)
    
    for t=1:size(Y)[2]
        for m=1:fhmm.M    
            for l=1:fhmm.M
                if l != m
                    Ỹ[:,m,t] += fhmm.W[:,:,m] * γ[m,:,t] 
                end
            end
            Ỹ[:,m,t] = Y[:,t] - Ỹ[:,m,t]
        end
    end
    display("C⁻¹")
    display(C⁻¹)
    display("Δ")
    display(Δ)
    display("Ỹ")
    display(Ỹ) 

    h_new = copy(h)
    for t=1:size(Y)[2]
        for m=1:fhmm.M
           h_new[:,t,m] = transpose(fhmm.W[:,:,m]) * C⁻¹ * Ỹ[:,m,t] - 0.5Δ[:,:,m]
        end
    end
    display("h pre exp")
    display(h_new)
    h_new = exp.(h_new) 
    #delta = sum(h_new - h)
    #display("Delta: $delta")
    return h_new
end

function m_step(fhmm::FHMM, 
    h::Array{Float64,3}, # (KxTxM) 
    Y::AbstractMatrix, # (DxT)
    γ::Array{Float64, 3},
    ξ::Array{Float64, 4}) # (M, K, K, T-1) 
    T = size(Y)[2]
    D = size(Y)[1]
    for m=1:fhmm.M
      for k=1:fhmm.K
        fhmm.π[k,m] = γ[m,k,1]
      end
    end
    
    for m=1:fhmm.M
      for k=1:fhmm.K
        for j=1:fhmm.K
          fhmm.P[k,j,m] = sum(ξ[m,k,j,:]) / sum(γ[m,j,1:end-1])
        end
      end
    end
            
    w_sum = zeros(fhmm.D,fhmm.D)
    for t=1:T
      for m=1:fhmm.M
        w_sum += fhmm.W[:,:,m] * γ[m,:,t] * Y[:,t]'
      end
    end
    
    fhmm.C[:,:] = (sum(Y * Y', dims=3) .* (1/T)) - (w_sum .* 1/T)
    
    St = reshape(γ, fhmm.M*fhmm.K, T)
    
    sum_Yt_St = zeros(D, fhmm.M*fhmm.K)
    for t=1:T
      sum_Yt_St[:,:] += Y[:,t] * transpose(St[:,t])
    end

    exp_StSt = zeros(fhmm.M*fhmm.K, fhmm.M*fhmm.K)
    
    for t=1:T
       exp_StSt[:,:] += St[:,t] * St[:,t]' 
    end
    
    fhmm.W[:,:,:] = sum_Yt_St * pinv(exp_StSt) # (DxMK)*(MKxMK)
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
    # Q = ones(fhmm.M, fhmm.K, T) # this is Q(Mt|Mt-1,theta)
    γ = ones(M, K, T) 
    ξ = zeros(K, K, T-1, M) 
    
    # initial values for variational parameter h (probability of a fictitious observation for every chain being in every state at every timestep)
    h = rand(K, T, M) 
    h ./= sum(h,dims=(1))
    
    @assert(all(isapprox.(sum(h, dims=(1)), 1)))
    
    # holder for likelihood so we can compare improvement across 
    #likelihood::Vector{Float64} = zeros(1)
    #for i=1:maxiter
    #    likelihood_i = compute_likelihood()
    #end
    
    for i in 1:10
      γ, ξ = forward_backward(fhmm, Y, h)
      display("γ")
      display(γ)
      display("ξ")
      display(ξ)
      h = update_variational_parameters(fhmm, h, Y, γ)
      display("h")
      display(h)
      m_step(fhmm, h, Y, γ, ξ)
      display("pi")
      display(fhmm.π)
      display("W")
      display(fhmm.W)
      display("P")
      display(fhmm.P)
      display("C")
      display(fhmm.C)
    end
end

