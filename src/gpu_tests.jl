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

l = [[1.0, 2.0], [3.0, 4.0]]
s = [5.0, 6.0, 7.0]
r = [[8.0, 9.0]]

set_backend(:cpu)
func(x::Vector{Float64}) = sum(x)

tensor = one_site_block(l, s, r, func)

set_backend(:gpu)

tensor = one_site_block(l, s, r, func)