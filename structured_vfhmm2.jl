using InvertedIndices
using Statistics 
using LinearAlgebra
using ProximalOperators

const ϵ = exp(-700);


function nearestSPD(A::AbstractMatrix)

    B = (A + A')/2;

    U,Sigma,V = svd(B);
    H = V*diagm(Sigma)*V';
  
    Ahat = (B+H)/2;
    
    Ahat = (Ahat + Ahat')/2;
  
    k = 0;
    #for i=1:size(A)[1]
    #  A[i,i] += ϵ
    #end
    while isposdef(Ahat) == false
      mineig = minimum(eigen(Ahat).values);
      Ahat = Ahat + (-mineig*k^2 + eps(mineig))*I;
      k += 1
      if k > 10000
        println("Unstable")
        break
      end
    end
    return Ahat
  end    

# calculate P(Mt|Yt,θ) for E_Mt
function forward_backward(
    model::StructuredFHMM)
    N = model.N
    T = model.T
    M = model.M
    γ = model.γ
    K = model.K

    scale = zeros(T*N,M)
    α = model.α
    exph =  exp.(model.h)    
    π = model.π
    β = model.β
    for m=1:M
      i_m = (m-1)*K+1:m*K; #d1
      i_k = 1:N; #d2
      
      # forward pass
      α[i_k,i_m] = exph[i_k,i_m] .* (ones(N,1) * π[:,m]')
      scale[i_k,m]=sum(α[i_k,i_m],dims=2) .+ ϵ
      rdiv!(α[i_k,i_m], scale[i_k,m][1])
      
      for t=2:T  
        i_t = (t-1)*N+1:t*N # d2
        α[i_t, i_m] = (α[i_t.-N, i_m]* model.P[i_m,:]) .* exph[i_t, i_m]
        scale[i_t,m] = sum(α[i_t, i_m], dims=2) .+ ϵ
        rdiv!(α[i_t,i_m], scale[i_t,m][1])  
      end
      
      # backward pass
      i_t = (T-1)*N+1:T*N #d2
      β[i_t, i_m] = ones(N,K)
      rdiv!(β[i_t, i_m], scale[i_t, m][1])
      for t=T-1:-1:1
        i_t=(t-1)*N+1:t*N # d2
        β[i_t, i_m] = (β[i_t.+N, i_m] .* exph[i_t.+N,i_m]) * model.P[i_m,:]'
        rdiv!(β[i_t, i_m], scale[i_t, m][1])
      end
      
      # calculate gamma
      γ[:,i_m] = α[:,i_m] .* β[:,i_m]
      γ[:,i_m] ./= sum(γ[:,i_m], dims=2) # CHECK
      γ[:,i_m] .+= ϵ
    end

end

function update_variational_parameters(
    model::StructuredFHMM, 
    Yalt::AbstractMatrix) # TN*p

    C⁻¹::Matrix{Float64} = model.C^-1    
    N = model.N
    h = model.h
    M = model.M
    K = model.K
    
    for t=1:model.T
      i_s = (t-1)*N+1:t*N # the index of the step in the sequence
      Ỹ=model.γ[i_s,:] * model.W
      for m=1:M
        i_w=(m-1)*K+1:m*K # the index of this chain's weights
        Wm = model.W[i_w,:]
        Ci = C⁻¹
        #WmCi = Wm * C⁻¹
        #WmCiWm = WmCi * Wm'
        #h[i_s,i_w] = (
        #  WmCi * ((Yalt[i_s,:] - Ỹ)') + 
         # (WmCiWm * model.γ[i_s, i_w]') - 
         # 0.5 .* (diag(WmCiWm) * ones(1,N)))'
        h[i_s,i_w] = (
          (Wm * Ci * ((Yalt[i_s,:] - Ỹ)')) + 
          (Wm * Ci * Wm' * model.γ[i_s, i_w]') - 
          (0.5 .* (diag(Wm * Ci * Wm') * ones(1,N))))'
         h[i_s,i_w] .-= (maximum(h[i_s,i_w]', dims=1)' * ones(1,K))
      end
    end
    
    @assert(!any(isinf.(model.h)))    
    @assert(!any(isnan.(model.h)))    
end

function m_step(
    model::StructuredFHMM, 
    Yalt::AbstractMatrix,
    YY::AbstractMatrix)
    exph = exp.(model.h)
    
    T = model.T
    N = model.N
    D = model.D
    K = model.K
    M = model.M
    γ = model.γ
    η = model.γ' * model.γ;
    ξ = model.ξ
    α = model.α
    β = model.β
    
    γsum = sum(γ, dims=1)
    for m=1:M
      i = (m-1)*K+1:m*K;
      η[i,i] = diagm(γsum[i])
    end

    γY = γ' * Yalt;
    @assert(!any(isnan.(γY)))    
    
    η = (η + η') ./ 2
    fill!(model.ξ, 0)
    for t=1:T-1
      i_t=(t-1)*N+1:t*N
      i_t1=i_t.+N
      for m=1:M
        m_n=(m-1)*K+1:m*K;
        t = model.P[m_n,:] .* (α[i_t, m_n]' * (β[i_t1,m_n] .* exph[i_t1, m_n]))
        model.ξ[m_n,:] .+= (t / (sum(t, dims=1)))
      end
    end
    
    @assert(!any(isnan.(model.ξ)))
   
    # update parameters
    U,S,V = svd(nearestSPD(η))

    Si=zeros(K*M, K*M)
    
    for mk=1:K*M
      if S[mk] < maximum(size(S)) * norm(S) * 0.001
        Si[mk,mk] = 0
      else
        Si[mk,mk] = 1/S[mk]
      end
    end
    
    model.W[:,:,:] = V * Si * U' * γY
    
    model.C[:,:] = YY - γY' * pinv(η, atol=M*K*eps(norm(η))) * γY / (N*T);
    model.C[:,:] = nearestSPD(model.C + model.C' / 2)
    
    dC = det(model.C)
    
    if(dC < 0)
      println("Ill-conditioned covariance matrix, aborting m-step");
      return
    end
      
    for mk=1:K*M
      sumξ = sum(ξ[mk,:],dims=1)
      if(sumξ == 0)
        model.P[mk,:] = ones(1,K) / K;
      else
        model.P[mk,:] = ξ[mk,:] / sumξ
      end
    end
    model.π[:,:] = reshape(sum(model.γ[1:N,:], dims=1), K,M) / N;
end

function base(k::Int, m::Int, d::Int)
  mm=m.^(d-1:-1:0)
  v = ones(Int, d)
  for i=1:d
    v[i]=Int(floor.(k/mm[i]))
    k=k-mm[i]*v[i]
  end
  return v .+ 1
end

# this is calculating Q({St}) log [ P({St, Yt}) / Q({St})
function compute_likelihood(
  model::StructuredFHMM, 
  Yalt::AbstractMatrix)
  
  K = model.K
  M = model.M
  T = model.T
  D = model.D
  N = model.N
  γ = model.γ

  k1 = model.k1 
   
  dd = zeros(Int, K^M, M)
  
  Wb = zeros(K^M, D)
  γ_tmp = ones(T*N, K^M)
  
  for i=1:K^M
    dd[i,:] = base(i-1, K, M)
    for m=1:M
        Wb[i,:] += model.W[(m-1)*K + dd[i,m],:]
        γ_tmp[:,i] .*= γ[:, (m-1)*K + dd[i,m]]
    end
  end
  
  ll::Float64 = 0.0
  logγ = log.(γ)
  logP = log.(model.P)
  logPi = log.(model.π)
  
  C⁻¹ = model.C^-1
    
  d = det(model.C)
  
  if d < 0
    display("Ill-conditioned covariance matrix, aborting");
    display(model.C)
    return ll;
  end

  k2=model.k1/(sqrt(d))
  
  for l=1:K^M
    d = (ones(N*T, 1) * vec(Wb[l,:])') - Yalt
    ll -= 0.5 * sum(γ_tmp[:,l] .* sum((d * C⁻¹).*d,dims=2))
  end

  ll += T * N * log(k2)
  
  ll += sum(γ[1:N,:] * vec(logPi)) - sum(γ[1:N,:] .* logγ[1:N,:])

  for t=2:T 
    i_t1 = (t-1)*N+1:t*N
    i_t0 = (t-2)*N+1:(t-1)*N
    for m=1:M
      i_t2 = (m-1)*K+1:m*K
      ll += sum(
          model.γ[i_t0, i_t2] .* (model.γ[i_t1, i_t2] * logP[i_t2,:]')) - 
        sum(
          model.γ[i_t1, i_t2] .* logγ[i_t1, i_t2])
    end
  end

  return ll
end
    
function fit_structured!(model::StructuredFHMM,
                 Y::AbstractMatrix, # shape (N*T, D)
                 iter_v::Int=10,
                 iter_bw::Int=100,
                 tol::Float64=1.0e-6,
                 ll_every::Int=5)
    N = model.N
    T = model.T
    # reshape Y to TN*p rather than NT*p
    Yalt = copy(Y)
    for t=1:model.T
        for n=1:model.N
          Yalt[((t-1)*N)+n,:] = Y[((n-1)*T)+t,:]
        end
    end
    YY=Y'*Y/(N*T) 
      
    ll = 0
    ll_base = 0
    ll_old = 0
    γ = model.γ
    
    logγ_old = copy(γ)
    
    
    
    for i in 1:iter_bw
      for j in 1:iter_v
        update_variational_parameters(model, Yalt)
        forward_backward(model)
        γc = copy(γ)
        γc[findall(x->x == 0,γc)]  .= ϵ
        logγ = log.(γ)
        if j > 1
          γdiff = sum(model.γ .* logγ) - sum(model.γ .* logγ_old)
          if γdiff < N*T*tol
            display("Variational converged in $j iterations")
            break;
          end
        end
        logγ_old[:,:] = logγ
      end
      display("step $i: log likelihood $ll")
      ll_old = ll
      ll = compute_likelihood(model, Yalt)
      if i <= 2
        ll_base = ll
      elseif(i % ll_every == 0 &&  (ll - ll_base) < (1 + tol) * (ll_old - ll_base))
        display("Converged after $i iterations")
        return
      end      
      m_step(model, Yalt,YY)
    end
end

