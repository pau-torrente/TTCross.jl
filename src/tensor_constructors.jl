using IterTools
using .TTcross

function two_site_block(
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

    # TODO Combinations is an iterable that when mapped already gives a tensor -> Need to 
    # touch this in order to pass it onto the gpu
    
    combinations = product(i_k_1, s_k, s_kp1, j_kp1)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, li, ls_k, ls_kp1, lj)

    return tensor
end

function two_site_block(
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Vector{Vector{Float64}},
    func::Function,
)

    ls_k = length(s_k)
    ls_kp1 = length(s_kp1)
    lj = length(j_kp1)

    combinations = product(i_k_1, s_k, s_kp1, j_kp1)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, 1, ls_k, ls_kp1, lj)

    return tensor
end

function two_site_block(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    func::Function,
)

    li = length(i_k_1)
    ls_k = length(s_k)
    ls_kp1 = length(s_kp1)

    combinations = product(i_k_1, s_k, s_kp1)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, li, ls_k, ls_kp1, 1)

    return tensor
end

function one_site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func::Function,
)
    li = length(i_k_1)
    ls_k = length(s_k)
    lj = length(j_k)

    combinations = product(i_k_1, s_k, j_k)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, li, ls_k, lj)

    return tensor
end

function one_site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    func::Function,
)
    li = length(i_k_1)
    ls_k = length(s_k)

    combinations = product(i_k_1, s_k)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, li, ls_k, 1)

    return tensor
end

function one_site_block(
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func::Function,
)
    ls_k = length(s_k)
    lj = length(j_k)

    combinations = product(s_k, j_k)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    tensor = reshape(results, 1, ls_k, lj)

    return tensor
end

function inverse_block(
    i_k::Vector{Vector{Float64}},
    j_k::Vector{Vector{Float64}},
    func::Function
)
    li = length(i_k)
    lj = length(j_k)

    combinations = product(i_k, j_k)

    if _backend == :cpu
        results = map((x) -> func(x...), combinations)
    elseif _backend == :gpu
        combinations = CuArray(combinations)
        results = Array(gpu_ten_constructor(combinations, func))
    end

    matrix = reshape(results, li, lj)
    # TODO Check if CUDA.reclaim() is needed to free up the memory on the gpu
    # TODO move this inversion also to the gpu, if selected
    return inv(matrix)
end

