using IterTools
using CUDA

# TODO All functions could share the same name and leverage the multiple dispatch feature of Julia

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
    
    combinations = product(i_k_1, s_k, s_kp1, j_kp1)

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat(a..., [b], [c], d...) for (a, b, c, d) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, li, ls_k, ls_kp1, lj)
    end

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
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat([a], [b], c...) for (a, b, c) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, 1, ls_k, ls_kp1, lj)
    end
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
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat(a..., [b], [c]) for (a, b, c) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, li, ls_k, ls_kp1, 1)
    end

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
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat(a..., [b], c...) for (a, b, c) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, li, ls_k, lj)
    end

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
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat(a..., [b]) for (a, b) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, li, ls_k, 1)
    end
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
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat([a], b...) for (a, b) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(results, 1, ls_k, lj)
    end

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
        matrix = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        resh_combinations = stack([vcat(a..., b...) for (a, b) in vec(collect(combinations))])
        cu_combs = CuArray(resh_combinations)
        results = Array(gpu_ten_constructor(cu_combs, func))
        matrix = reshape(results, li, lj)
    end

    # TODO Check if CUDA.reclaim() is needed to free up the memory on the gpu
    # TODO move this inversion also to the gpu, if selected
    return inv(matrix)
end

