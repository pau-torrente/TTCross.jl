using IterTools

# TODO Move the CPU map to a separate function, leave these methods as generic and move the mapping to the seggregated corresponding functions
function _cpu_2site_block(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Vector{Vector{Float64}},
    func::Function,
)

    li = length(i_k_1)
    ls_k = length(s_k)
    ls_kp1 = length(s_kp1)
    lj = length(j_kp1)

    tensor = Array{Float64}(undef, li, ls_k, ls_kp1, lj)

    combinations = product(i_k_1, s_k, s_kp1, j_kp1)

    # THIS IS THE ONLY PART THAT DIFFERS FROM THE GPU IMPLEMENTATION
    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, li, ls_k, ls_kp1, lj)

    return tensor
end

function _cpu_2site_block(
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Vector{Vector{Float64}},
    func::Function,
)

    ls_k = length(s_k)
    ls_kp1 = length(s_kp1)
    lj = length(j_kp1)

    combinations = product(s_k, s_kp1, j_kp1)

    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, 1, ls_k, ls_kp1, lj)

    return tensor
end

function _cpu_2site_block(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    func::Function,
)

    li = length(i_k_1)
    ls_k = length(s_k)
    ls_kp1 = length(s_kp1)

    combinations = product(i_k_1, s_k, s_kp1)

    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, li, ls_k, ls_kp1, 1)

    return tensor
end

function _cpu_1site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func::Function,
)
    li = length(i_k_1)
    ls_k = length(s_k)
    lj = length(j_k)

    combinations = product(i_k_1, s_k, j_k)

    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, li, ls_k, lj)

    return tensor
end

function _cpu_1site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    func::Function,
)
    li = length(i_k_1)
    ls_k = length(s_k)

    combinations = product(i_k_1, s_k)

    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, li, ls_k, 1)

    return tensor
end

function _cpu_1site_block(
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func::Function,
)
    ls_k = length(s_k)
    lj = length(j_k)

    combinations = product(s_k, j_k)

    results = map((x) -> func(x...), combinations)

    tensor = reshape(results, 1, ls_k, lj)

    return tensor
end

function _cpu_inverse_block(
    i_k::Vector{Vector{Float64}},
    j_k::Vector{Vector{Float64}},
    func::Function
)
    li = length(i_k)
    lj = length(j_k)

    combinations = product(i_k, j_k)

    results = map((x) -> func(x...), combinations)

    matrix = reshape(results, li, lj)

    return tensor
end

