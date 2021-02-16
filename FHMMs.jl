module FHMMs

# Factorial Hidden Markov Models (FHMM)

# Reference:
# [Ghahramani1997] Ghahramani, Zoubin, and Michael I. Jordan.
# "Factorial hidden Markov models."
# Machine learning 29.2-3 (1997): 245-273.

export FHMM, fit!, fit_sv!, update_variational_parameters

# 
# Gaussian Factorial HMM (FHMM)
# At timestep T, the model will emit a vector of size D drawn from a Gaussian with a mean determined by the state vector K (sum of state vectors M) and covariance C
#
struct FHMM
    D::Int # observation dimensions
    K::Int # number of states
    M::Int # number of factors
    # contributions to the means of Gaussian, shape (D, K, M)
    W::Array{Float64, 3}    
    # initial state probability, shape (K, M)
    π::Array{Float64, 2}
    # transition matrices, shape (K, K, M)
    P::Array{Float64, 3}
    # covariance matrix, shape (D, D)
    C::Array{Float64, 2}

    function FHMM(D::Int, K::Int, M::Int)
        C = let X = randn(D, D); X * X' + I; end
        new(D, K, M,
            rand(D, K, M),
            ones(K, M) ./ (K*M),
            ones(K, K, M) ./ K,
            C
            )
    end
    function FHMM(D::Int, K::Int, M::Int, W::Array{Float64,3},  π::Array{Float64, 2}, P::Array{Float64, 3},C::Array{Float64, 2})
        @assert(size(W)[1] == D)
        @assert(size(W)[2] == K)
        @assert(size(W)[3] == M)
        @assert(size(π)[1] == K)
        @assert(size(π)[2] == M)
        @assert(size(P)[1] == K)
        @assert(size(P)[2] == K)
        @assert(size(P)[3] == M)
        @assert(size(C)[1] == D)
        @assert(size(C)[2] == D)
        new(D, K, M, W, π, P, C)
    end
    
end



for fname in [  "completely_factorized_vfhmm", "structured_vfhmm", "structured_vfhmm2" ]
    include(string(fname, ".jl"))
end

end # module
