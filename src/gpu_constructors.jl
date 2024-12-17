#TODO Check what is going on with float types from and to the gpu
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
    
    kern = @cuda launch=false tenconstr_kernel(combinations, output, func)
    config = launch_configuration(kern.fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)

    kern(combinations, output; threads = threads, blocks = blocks)

    return output  
end