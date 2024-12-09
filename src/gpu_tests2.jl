using CUDA
using IterTools
using LinearAlgebra

function f(x)
    out = 0.0
    for xi in x
        out += xi
    end
    return out
end

f([1.0, 2.0, 3.0])
function tenconstr_kernel(combinations::CuDeviceArray{Float64}, out::CuDeviceVector{Float64}, func)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(combinations)[2]
        out[i] = func(@view combinations[:, i])
    end
    return nothing
end

function gpu_ten_constructor(combinations::CuArray, func)
    output = CUDA.zeros(Float64, size(combinations)[2])

    len = size(combinations)[2]
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
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func,
)
    li = length(i_k_1)
    ls_k = length(s_k)
    lj = length(j_k)

    t1 = time()
    combinations = product(i_k_1, s_k, j_k)
    println("Time to compute combinations: ", time() - t1)

    t1 = time()
    resh_combinations = stack([vcat(a..., [b], c...) for (a, b, c) in vec(collect(combinations))])
    println("Time to stack combinations: ", time() - t1)

    t1 = time()
    cu_combs = CuArray(resh_combinations)
    println("Time to convert to CuArray: ", time() - t1)

    t1 = time()    
    results = Array(gpu_ten_constructor(cu_combs, func))
    println("Time to compute results: ", time() - t1)

    t1 = time()
    tensor = reshape(results, li, ls_k, lj)
    println("Time to reshape: ", time() - t1)

    return tensor
end

n_samples = 600
l = [[1.0, 2.0] for _ in 1:n_samples]
s = [5.0, 6.0, 7.0]
r = [[8.0, 9.0] for _ in 1:n_samples]

tensor = one_site_block(l, s, r, f)


function one_site_block2(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    j_k::Vector{Vector{Float64}},
    func::Function,
)
    li = length(i_k_1)
    ls_k = length(s_k)
    lj = length(j_k)

    t1 = time()
    combinations = product(i_k_1, s_k, j_k)
    println("Time to compute combinations: ", time() - t1)

    t1 = time()
    tensor = map((x) -> func(vcat(x...)), combinations)
    println("Time to compute tensor: ", time() - t1)

    return tensor
end

tensor2 = one_site_block2(l, s, r, f)

# Kernel that constructs the combinations on the fly and applies the function
function tenconstr_kernel_no_combs(
    i_k_1::CuDeviceArray{Float64}, 
    s_k::CuDeviceVector{Float64}, 
    j_k::CuDeviceArray{Float64}, 
    out::CuDeviceVector{Float64}, 
    func,
    li::Int, ls_k::Int, lj::Int)
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= li * ls_k * lj
        # Compute the corresponding indices in i_k_1, s_k, and j_k
        s_idx = div(i, li * lj) + 1
        j_idx = div(i % (li * lj), li) + 1
        i_idx = i % li + 1
        
        # Get the combination of vectors (from i_k_1, s_k, j_k) based on the indices
        combination = vcat(
            i_k_1[:, i_idx],
            s_k[s_idx],
            j_k[:, j_idx]
        )
        
        # Apply the function on the combination
        out[i] = func(combination)
    end
    return nothing
end

# GPU tensor constructor that avoids pre-stacked combinations
function gpu_ten_constructor_no_combs(
    i_k_1::CuArray, s_k::CuArray, j_k::CuArray, func)
    
    li = size(i_k_1, 2)
    ls_k = length(s_k)
    lj = size(j_k, 2)
    
    len = li * ls_k * lj
    output = CUDA.zeros(Float64, len)
    
    kern = @cuda launch=false tenconstr_kernel_no_combs(i_k_1, s_k, j_k, output, func, li, ls_k, lj)
    config = launch_configuration(kern.fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)

    kern(i_k_1, s_k, j_k, output, func, li, ls_k, lj; threads=threads, blocks=blocks)
    
    return output
end

# Main function that handles the tensor construction
function one_site_block_no_combs(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    j_k::Vector{Vector{Float64}}, 
    func)
    
    li = length(i_k_1)
    ls_k = length(s_k)
    lj = length(j_k)

    # Convert inputs to CuArrays
    t1 = time()
    cu_i_k_1 = CuArray(hcat(i_k_1...))
    cu_s_k = CuArray(s_k)
    cu_j_k = CuArray(hcat(j_k...))
    println("Time to convert inputs to CuArray: ", time() - t1)

    # Call the GPU tensor constructor
    t1 = time()
    results = Array(gpu_ten_constructor_no_combs(cu_i_k_1, cu_s_k, cu_j_k, func))
    println("Time to compute results: ", time() - t1)

    # Reshape the results into a tensor
    t1 = time()
    tensor = reshape(results, li, ls_k, lj)
    println("Time to reshape: ", time() - t1)

    return tensor
end

# Example usage
n_samples = 600
l = [[1.0, 2.0] for _ in 1:n_samples]
s = [5.0, 6.0, 7.0]
r = [[8.0, 9.0] for _ in 1:n_samples]

tensor = one_site_block_no_combs(l, s, r, f)
