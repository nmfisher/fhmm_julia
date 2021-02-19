module FHMMs

# Factorial Hidden Markov Models (FHMM)

# Reference:
# [Ghahramani1997] Ghahramani, Zoubin, and Michael I. Jordan.
# "Factorial hidden Markov models."
# Machine learning 29.2-3 (1997): 245-273.

export fit_structured, base

struct StructuredFHMM
  h::Array{Float64,2} # (NT, MK) the variational parameter (internal only)
  # the forward/backward parameters
  γ::Array{Float64, 2} # (NT, MK)
  ξ::Array{Float64, 2} # (MT, K)
  α::Array{Float64, 2}
  β::Array{Float64, 2}
  k1::Float64
  Y::AbstractMatrix # (NT, D)
  D::Int # observation dimensions
  K::Int # number of states
  M::Int # number of factors
  T::Int # number of timesteps
  N::Int # batch size
  # contributions to the means of Gaussian, shape (MK, D)
  W::Array{Float64, 2}    
  # initial state probability, shape (K, M)
  π::Array{Float64, 2}
  # transition matrices, shape (KM, M)
  P::Array{Float64, 2}
  # covariance matrix, shape (D, D)
  C::Array{Float64, 2}
  
  function StructuredFHMM(
      Y::AbstractMatrix, 
      M::Int, 
      K::Int, 
      D::Int,
      T::Int,
      N::Int)
      
      # initialize covariance matrix from observations
      C = diagm(diag(cov(Y)))
      
      # initialize weights (aka μ) from mean of observations
      W = rand(M*K, D) * sqrt(C)/M + ones(K*M, 1) * mean(Y,dims=1)/M
      
      # initialize initial state probabilities
      π = rand(K,M)
      π ./= sum(π, dims=1)
      
      # initialize transition state probabilities
      P=rand(K*M,K)
      P ./= sum(P, dims=2)
      
      # initialize variational parameter
      h = ones(N*T, M*K) /K
      h ./= sum(h,dims=(1))
      
      # initialize forward-backward parameters
      γ = ones(N*T, M*K)
      ξ = zeros(M*K, K)
      
      # these are internal only, we just preallocate for efficiency
      α = zeros(N*T, M*K)
      β = zeros(N*T, M*K)
      
      # constant for normal PDF
      k1=(2*pi)^(-D/2)

      new(
        h, 
        γ, 
        ξ, 
        α, 
        β, 
        k1, 
        Y, 
        D, 
        K, 
        M, 
        T, 
        N, 
        W, 
        π, 
        P, 
        C)
  end
end


for fname in [  "structured_vfhmm2" ]
    include(string(fname, ".jl"))
end

end # module
