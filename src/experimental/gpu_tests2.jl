# using CUDA
# using IterTools
# using LinearAlgebra
using TTcross
using CUDA
using SplitApplyCombine

function f(x)
    out = 0.0
    for xi in x
        out += xi
    end
    return out
end

f([1.0, 2.0, 3.0])

function product_to_cuarray(iterator::Base.Iterators.ProductIterator)
    collected = combinedims(map(x -> vcat(x...), collect(iterator)))
    shape = size(collected)
    return cu(reshape(collected, (shape[1], reduce(*, shape[2:end])))), shape
end

function tenconstr_kernel(combinations::CuDeviceArray{Float32}, out::CuDeviceVector{Float32}, func)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(combinations)[2]
        @inbounds out[i] = func(@view combinations[:, i])
    end
    return nothing
end

function tenconstr_kernel(combinations::CuDeviceArray{Float32}, out::CuDeviceVector{Float64}, func)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(combinations)[2]
        @inbounds out[i] = func(@view combinations[:, i])
    end
    return nothing
end

function gpu_ten_constructor(combinations::CuArray, func)
    
    len = size(combinations)[2]

    output = CUDA.zeros(Float64, len)

    println("Len: ", len)
    
    kern = @cuda launch=false tenconstr_kernel(combinations, output, func)
    config = launch_configuration(kern.fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)

    println("Threads: ", threads)
    println("Blocks: ", blocks)

    kern(combinations, output; threads = threads, blocks = blocks)

    return output  
end


function one_site_block(
    i_k_1::Matrix{Float64},
    s_k::Vector{Float64},
    j_k::Matrix{Float64},
    func::Function,
)
    t1 = time()
    combinations = Base.product(eachcol(i_k_1), s_k, eachcol(j_k))
    println("combinations ", time() - t1)

    if _backend == :cpu
        t1 = time()
        tensor = map((x) -> func(vcat(x...)), combinations)
        print("cpu map ", time() - t1)
        println("")

    elseif _backend == :gpu
        t1 = time()
        cu_combs, shape = product_to_cuarray(combinations)
        println("gpu combinations", time() - t1)
        println("")

        t1 = time()
        array = Array(gpu_ten_constructor(cu_combs, func))
        println("gpu computation", time() - t1)
        tensor = reshape(array, shape[2:end])
        println("")

    end

    return tensor
end

global _backend = :cpu
n_samples = 200
l = hcat([[1.0, 2.0] for _ in 1:n_samples]...)
s = [5.0, 6.0, 7.0]
r = hcat([[8.0, 9.0] for _ in 1:n_samples]...)

tensor = one_site_block(l, s, r, f)
global _backend = :gpu

# set_backend(:gpu)

tensor = one_site_block(l, s, r, f)