using InvertedIndices
using Statistics 

const ϵ = 1.0e-64

# we need to calculate P(Mt|Yt,θ) for E_Mt
function forward_backward(
    fhmm::FHMM, 
    Y::AbstractMatrix, # (DxT),
    h::Array{Float64,3}, # (KxTxM) )
    γ::Array{Float64,3}, # MxKxT
    ξ::Array{Float64,4}) # MxKxKxT-1
    @assert(!any(isnan.(fhmm.P)))
    
    T=size(Y)[2]
    K = fhmm.K
    scale = zeros(fhmm.M,T)

    α = zeros(fhmm.K, T, fhmm.M)
    β = zeros(fhmm.K, T, fhmm.M)
    
    exp_s = zeros(fhmm.M, fhmm.K, fhmm.K, T-1)
    
    # forward recursion
    for m=1:fhmm.M
      #display("m = $m")
      #display("h")
      #display(h[:,1,m])
      #display("pi")
      #display(fhmm.π[:,m])
      α[:,1,m] = h[:,1,m] .* fhmm.π[:,m]
      scale[m,1] = sum(α[:,1,m])
      α[:,1,m] ./=scale[m,1]
      #display("α[:,1,m]")
      #display(α[:,1,m])
    end
    

    
    for t=2:T
      for m=1:fhmm.M
        α[:,t,m] = h[:,t,m]' .* (α[:,t-1,m]' * fhmm.P[:,:,m])
        scale[m,t] = sum(α[:,t,m])
        α[:,t,m] ./= scale[m,2]
      end
    end
        
    #display("α")
    #display(α)
    
    
    β[:,T,:] .= 1 / K
    
    # backwards
    for t=T-1:-1:1
      for m = 1:fhmm.M
        #display("m = $m, t = $t")
        #display("β[:,t+1,m]")
        #display(β[:,t+1,m])
        #display("fhmm.P[:,:,m]")
        #display(fhmm.P[:,:,m])
        #display("h[:,t,m]")
        #display(h[:,t,m])
        β[:,t,m] = β[:,t+1,m]' * fhmm.P[:,:,m] .* h[:,t+1,m]'
        β[:,t,m] ./= sum(β[:,t,m])
      end
    end
    
    #display("β")
    #display(β)
    
    @assert(!any(isnan.(β)))
    @assert(!any(isnan.(α)))


    for t=1:T
      for m=1:fhmm.M
        γ[m,:,t] = (α[:,t,m] .* β[:,t,m]) 
      end
    end
    for t=1:T-1
      for m=1:fhmm.M
        for k=1:fhmm.K
          for j=1:fhmm.K
            #display("t $t m $m k $k j $j")
            #display(α[k,t,m])
            #display(fhmm.P[k,j,m])
            #display(β[j,t+1,m])
            #display(h[j,t+1,m])
            ξ[m,k,j,t] = α[k,t,m] * fhmm.P[k,j,m] * β[j,t+1,m] * h[j,t+1,m]
          end
        end
      end
    end
    γ .+= ϵ
    γ[:,:,:] ./= sum(γ[:,:,:], dims=2) 
    
    #  ξ[m,:,:,t] ./= sum(ξ[m,:,:,t])

    #display("γ")
    #display(γ)
    #display("ξ")
    #display(ξ)
    @assert(!any(isnan.(γ)))
    @assert(!any(isnan.(ξ)))
    return γ, ξ
end

function update_variational_parameters(
    fhmm::FHMM, 
    h::Array{Float64,3}, # (KxTxM) 
    Y::AbstractMatrix, # (DxT)
    γ::Array{Float64, 3}) # (MxKxT)

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    
    #display("C⁻¹")
    #display(C⁻¹)

    T=size(Y)[2]
    K = fhmm.K
    D = fhmm.D
    M = fhmm.M
    
    h_new = copy(h)
    
    WmCi = zeros(K,D,M)
    for m=1:M
      WmCi[:,:,m] = fhmm.W[:,:,m]' * C⁻¹
    end
    
    Δ = zeros(fhmm.K, fhmm.M)
    
    for m=1:fhmm.M
        #display("m = $m")
        WmCiWm = WmCi[:,:,m] * fhmm.W[:,:,m]
        #display("WmCi * Wm")
        #display(WmCiWm)
        Δ[:,m] = diag(WmCiWm)
    end
    
    #display("Δ")
    #display(Δ)
    
    Ỹ = zeros(fhmm.D,fhmm.M,T)
    
    for t=1:T
        for m=1:fhmm.M    
            for l=1:fhmm.M
                if l != m
                    #display("t $t m $m l $l")
                    #display("W")
                    #display(fhmm.W[:,:,l])
                    #display("gamma")
                    #display(γ[l,:,t])
                    Ỹ[:,m,t] += fhmm.W[:,:,l] * γ[l,:,t] 
                end
            end
            Ỹ[:,m,t] = Y[:,t] - Ỹ[:,m,t]
        end
    end
    
    #display("Ỹ")
    #display(Ỹ)
   
    for t=1:size(Y)[2]
        for m=1:fhmm.M
          #display("m = $m, t = $t")
          #display("WmCi[:,:,m]")
          #display(WmCi[:,:,m])
          #display(" Ỹ[:,m,t]")
          #display(Ỹ[:,m,t])
          ##display("WmCi[:,:,m] * Ỹ[:,m,t]")
          #display(WmCi[:,:,m] * Ỹ[:,m,t] )
          #display("Δ[:,m]")
          #display(Δ[:,m])
          h_new[:,t,m] = WmCi[:,:,m] * Ỹ[:,m,t] - 0.5Δ[:,m]
        end
    end
    #display("h new")
    #display(h_new)
    
    #display("max at each timestep")
    #display(maximum(h_new, dims=(1)))
    h_new .-= maximum(h_new, dims=1)
    
    #display("h new after subtracting max")
    #display(h_new)
    
    h_new[:,:,:] = exp.(h_new) 
    
    #display("h new after exp")
    #display(h_new)
    @assert(!any(isnan.(h_new)))
    @assert(!any(isinf.(h_new)))
    return h_new
    
end

function m_step(fhmm::FHMM, 
    h::Array{Float64,3}, # (KxTxM) 
    Y::AbstractMatrix, # (DxT)
    γ::Array{Float64, 3}, # MxKxT
    ξ::Array{Float64, 4}) # (M, K, K, T-1) 
    T = size(Y)[2]
    D = size(Y)[1]
    K = fhmm.K
    M = fhmm.M
    
    Yγ = zeros(D,K,M)
    γγ = zeros(K,K,M)
    
    for m=1:M
      Yγ[:,:,m] = Y * γ[m,:,:]'
      γγ[:,:,m] = γ[m,:,:] * γ[m,:,:]'
      γγ[:,:,m] += γγ[:,:,m]'
      γγ[:,:,m] ./= 2
    end
    
    #display("Yγ")
    #display(Yγ)
    #display("γγ")
    #display(γγ)
    W_new = copy(fhmm.W)
    for m=1:M
      U,S,V = svd(γγ[:,:,m])
      #display(U)
      #display(S)
      #display(V)
      #display(pinv(γγ[:,:,m]))
      W_new[:,:,m] = Yγ[:,:,m] * pinv(γγ[:,:,m])
    end
    #display("fhmm.W")
    #display(fhmm.W)
    
    fhmm.π[:,:] = γ[:,:,1]'
    
    for m=1:fhmm.M
        for j=1:fhmm.K
          for k=1:fhmm.K
            # is j-> the right order?
            fhmm.P[j,k,m] = sum(ξ[m,j,k,:]) / sum(γ[m,j,1:end-1])
        end
      end
    end
    
    #display("fhmm.P")
    #display(fhmm.P)
                               
    YY=Y*Y'./ T;    
    #display("YY")
    #display(YY)
    WγY = zeros(D,D)
    for t=1:T
      for m=1:M
        #display("t $t m $m")
        #display("fhmm.W[:,:,m]")
        #display(fhmm.W[:,:,m])
        #display("γ[m,:,t]")
        #display(γ[m,:,t])
        #display("fhmm.W[:,:,m] * γ[m,:,t]")
        #display(fhmm.W[:,:,m] * γ[m,:,t])
        #WγY += fhmm.W[:,:,m] * γ[m,:,t] * Y[:,t]'
        GammaX = γ[m,:,t] * Y[:,t]'
        #display(sum(γ[m,:,t] * γ[m,:,t]'), dims=1)
        eta = diagm(diag(sum(γ, dims=(1,3))))
        eta .+= eta'
        eta ./= 2
        eta_inv = pinv(eta)
        display(eta_inv)
        WγY += GammaX' * eta_inv * GammaX
      end
    end
    WγY ./= T
    
    fhmm.C[:,:] = YY - WγY
    
    #display("fhmm.C")
    #display(fhmm.C)
    dCov = det(fhmm.C)
    if(dCov < 0)
      #display(fhmm.C)
      display("Ill-conditioned covariance matrix [ det : $dCov ]")
    end
    #W_diff = norm(W_new) - norm(fhmm.W[:,:,:])
    #display("W_diff $W_diff")
    fhmm.W[:,:,:] = W_new

end

# this is calculating Q({St}) log [ P({St, Yt}) / Q({St})
function compute_likelihood(
  fhmm::FHMM, 
  Y::AbstractMatrix,
  γ::Array{Float64,3}, #MxKxT
  )
  D = fhmm.D
  T = size(Y)[2]
  K = fhmm.K
  M = fhmm.M
  k1=(2*pi)^(-D/2);
  likelihood::Float64 = 0.0
  logγ::Array{Float64, 3} = log.(γ .+ ϵ)
  logP::Array{Float64, 3} = log.(fhmm.P .+ ϵ)
  logPi::Array{Float64, 2} = log.(fhmm.π .+ ϵ)
  
  γF = ones(K,M,T)
    
  C⁻¹::Matrix{Float64} = fhmm.C^-1
  
  d = det(fhmm.C)
  k2=k1/sqrt(det(d));
  if d < 0
    display("Ill-conditioned covariance matrix")
    return
  end
  
  for t=1:T
    for m=1:M
      for k=1:K
        d = fhmm.W[:,k,m] - Y[:,t] # dx1
        s = d' * C⁻¹ # 1xD * DxD
        s .*= d' 
        likelihood -= 0.5 * sum(γ[m,k,t] * sum(s,dims=1))
      end
    end
  end
  
  likelihood *= T * log(k2)
  
  for m=1:fhmm.M
    for k=1:K
      likelihood += (γ[m,k,1] * logPi[k,m]) - (γ[m,k,1]'logγ[m,k,1])
    end
  end
      
  
  for t=2:T
    for m=1:M
      for k=1:K
        for j=1:K
          likelihood += γ[m,k,t-1] * logP[k,j,m] * γ[m,j,t] - (γ[m,j,t] .* logγ[m,j,t])
        end
      end
    end
  end

  @assert(!isnan(likelihood))
  return likelihood
end

function fit_sv2_start!(fhmm::FHMM,
                 Y::AbstractMatrix,
                 maxiter::Int=100,
                 tol::Float64=1.0e-5,
                 fzero::Bool=false,
                 verbose::Bool=false)
   D, T = size(Y)
    M = fhmm.M
    K = fhmm.K
    h = zeros(K,T,M)
    h[:,:,1] = [ 0.1 0.01; 0.05 0.2; 0.27 0.03 ]
    h[:,:,2] = [ 0.2 0.1; 0.3 0.1; 0.5 0.6 ]
    γ = zeros(fhmm.M, fhmm.K, T)
    γ[:,:,1] = [ 0.2 0.5 0.3; 0.8 0.1 0.1 ]
    γ[:,:,2] = [ 0.5 0.4 0.1; 0.2 0.2 0.6 ]
    
    ξ = zeros(fhmm.M, fhmm.K, fhmm.K, T-1)
    return fit_sv2!(fhmm, Y, h, γ, ξ, maxiter, tol, fzero, verbose)
end
    

function fit_sv2!(fhmm::FHMM,
                 Y::AbstractMatrix, # shape (D, T)
                 h::Array{Float64,3},
                 γ::Array{Float64,3},
                 ξ::Array{Float64,4},
                 maxiter::Int=100,
                 tol::Float64=1.0e-5,
                 fzero::Bool=false,
                 verbose::Bool=false)
    D, T = size(Y)
    M = fhmm.M
    K = fhmm.K
    
    # initial values for variational parameter h (probability of a fictitious observation for every chain being in every state at every timestep)
    #h = rand(K, T, M) 
    #h ./= sum(h,dims=(1))
    
   
    ll = 0
    
    for i in 1:100
      for j in 1:10
        h_old = copy(h)
        h = update_variational_parameters(fhmm, h, Y, γ)
        if(abs(norm(h) - norm(h_old)) < tol)
          #display("Variational params converged in $j iterations")
          break;
        end
        γ, ξ = forward_backward(fhmm, Y, h, γ, ξ)
      end
      
      #if i % 10 == 0
      ll_old = ll 
      ll = compute_likelihood(fhmm, Y, γ)
      if(ll + ll_old < log(tol))
        display("Converged after $i iterations")
        break
      end
      #end
      m_step(fhmm, h, Y, γ, ξ)
    end
    display(fhmm.C)
    return h, γ, ξ
end

