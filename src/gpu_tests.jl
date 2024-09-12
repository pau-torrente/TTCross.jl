using CUDA
using LinearAlgebra
using Distributed

@everywhere function integrator(vec)
    return π
end

a = ones(2000)
cua = CuArray(a)

b = ones(2000)
cub = CuArray(b)

c = ones(2000)
cuc = CuArray(c)

function kernel1!(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(a)
        c[i] = a[i] + b[i]
    end
    c[i] = a[i] + b[i]

    return nothing
end

function kernel2!(a, f, out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    out[i] = f(a[i])

    return nothing
end

kern = @cuda launch=false kernel1!(cua, cub, cuc)
config = launch_configuration(kern.fun)
threads = min(length(a), config.threads)
blocks = cld(length(a), threads)

kern(cua, cub, cuc; threads=threads, blocks=blocks)
synchronize()

secs = @elapsed kern(cua, cub, cuc; threads=threads, blocks=blocks)
println("GPU time: $secs seconds")

@elapsed cua .+ cub