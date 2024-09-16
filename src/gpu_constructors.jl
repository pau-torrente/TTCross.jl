using CUDA

function tenconstr_kernel!(combinations::CuArray, out::CuArray, func::Function)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(combinations)[1]
        out[i] = func(combinations[i, :])
    end    
    return nothing
end

function gpu_ten_constructor(combinations::CuArray, func::Function)::CuArray
    output = CUDA.zeros(size(combinations[1]))
    
    kern = @cuda launch=false tenconstr_kernel!(combinations, output, func)
    config = launch_configuration(kern.fun)
    threads = min(length(a), config.threads)
    blocks = cld(length(a), threads)

    kern(combinations, output, func; threads = threads, blocks = blocks)

    return output  
end