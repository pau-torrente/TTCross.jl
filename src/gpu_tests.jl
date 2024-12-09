using CUDA
using LinearAlgebra
using Distributed

function integrator(vec)
    return 3.2
end

a = ones(500000)
cua = CuArray(a)

b = ones(500000)
cub = CuArray(b)

c = ones(500000)
cuc = CuArray(c)

function kernel1!(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(a)
        c[i] = a[i] + b[i]
    end
    return nothing
end

function kernel2!(a, f, out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(a)
        out[i] = f(a[i])
    end
    return nothing
end

kern = @cuda launch=false kernel2!(cua, integrator, cuc)
config = launch_configuration(kern.fun)
threads = min(length(a), config.threads)
blocks = cld(length(a), threads)

kern(cua, integrator, cuc; threads=threads, blocks=blocks)
synchronize()

secs = @elapsed kern(cua, cub, cuc; threads=threads, blocks=blocks)
println("GPU time: $secs seconds")

@elapsed c = integrator.(a)

include("ttcross.jl")
using .TTcross
using CUDA
using IterTools


# TODO What about complex numbers?

function f!(x::T, out::V) where {T, V}
    s = size(x)[1]
    for i in 1:s
        out += x[i]
    end
end

function tenconstr_kernel!(combinations::CuDeviceArray{Float64}, out::CuDeviceVector{Float64}, func)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(combinations)[2]
        func(combinations[i, :], out[i])
    end    
end

function gpu_ten_constructor(combinations::CuArray, func)::CuArray
    output = CUDA.zeros(Float64, size(combinations)[2])

    len = size(combinations)[2]
    
    kern = @cuda launch=false tenconstr_kernel!(combinations, output, func)
    config = launch_configuration(kern.fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)

    kern(combinations, output; threads = threads, blocks = blocks)

    return output  
end

function one_site_block(
        i_k_1::Vector{Vector{Float64}},
        s_k::Vector{Float64},
        j_k::Vector{Vector{Float64}},
        func,
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

# function tenconstr_kernel!(combinations::CuDeviceArray{Float64}, out::CuDeviceVector{Float64}, func)
#     i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#     if i <= size(combinations)[2]
#         func(combinations[:, i], out[i])
#     end    
#     return nothing
# end

# function gpu_ten_constructor(combinations::CuArray, func::Function)::CuArray
#     output = CUDA.zeros(Float64, size(combinations)[2])

#     len = size(combinations)[2]
    
#     kern = @cuda launch=false tenconstr_kernel!(combinations, output, func)
#     config = launch_configuration(kern.fun)
#     threads = min(len, config.threads)
#     blocks = cld(len, threads)

#     kern(combinations, output; threads = threads, blocks = blocks)

#     return output  
# end


l = [[1.0, 2.0], [3.0, 4.0]]
s = [5.0, 6.0, 7.0]
r = [[8.0, 9.0]]

set_backend(:cpu)
func(x::Vector{Float64}) = sum(x)

tensor = one_site_block(l, s, r, func)

set_backend(:gpu)

tensor = one_site_block(l, s, r, f!)