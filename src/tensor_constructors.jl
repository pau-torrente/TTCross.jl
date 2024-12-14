using CUDA
using SplitApplyCombine

function product_to_cuarray(iterator::Base.Iterators.ProductIterator)
    collected = combinedims(map(x -> vcat(x...), collect(iterator)))
    shape = size(collected)
    return cu(reshape(collected, (shape[1], reduce(*, shape[2:end])))), shape
end

function two_site_block(
    i_k_1::Matrix{Float64}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Matrix{Float64},
    func::Function,
)
    combinations = Base.product(eachcol(i_k_1), s_k, s_kp1, eachcol(j_kp1))

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    return tensor
end

function two_site_block(
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Matrix{Float64},
    func::Function,
)
    combinations = Base.product(s_k, s_kp1, eachcol(j_kp1))

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    return tensor
end

function two_site_block(
    i_k_1::Matrix{Float64}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    func::Function,
)
    combinations = Base.product(eachcol(i_k_1), s_k, s_kp1)

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    return tensor
end

function one_site_block(
    i_k_1::Matrix{Float64},
    s_k::Vector{Float64},
    j_k::Matrix{Float64},
    func::Function,
)
    combinations = Base.product(eachcol(i_k_1), s_k, eachcol(j_k))

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        println(typeof(cu_combs))
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    return tensor
end

function one_site_block(
    i_k_1::Matrix{Float64},
    s_k::Vector{Float64},
    func::Function,
)
    combinations = Base.product(eachcol(i_k_1), s_k)

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end
    return tensor
end

function one_site_block(
    s_k::Vector{Float64},
    j_k::Matrix{Float64},
    func::Function,
)
    combinations = Base.product(s_k, eachcol(j_k))

    if _backend == :cpu
        tensor = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    return tensor
end

function inverse_block(
    i_k::Matrix{Float64},
    j_k::Matrix{Float64},
    func::Function
)
    combinations = Base.product(eachcol(i_k), eachcol(j_k))

    if _backend == :cpu
        matrix = map((x) -> func(vcat(x...)), combinations)
    elseif _backend == :gpu
        cu_combs, shape = product_to_cuarray(combinations)
        array = Array(gpu_ten_constructor(cu_combs, func))
        tensor = reshape(array, shape[2:end])
    end

    # TODO Check if CUDA.reclaim() is needed to free up the memory on the gpu
    # TODO move this inversion also to the gpu, if selected
    return inv(matrix)
end

