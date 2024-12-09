using CUDA

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

# TODO RIght now type is fixed to float -> Will give problems with complex functions
function os_kernel(
    i_k_1::CuDeviceArray{Float64},
    s_k::CuDeviceVector{Float64},
    j_k::CuDeviceArray{Float64},
    s1::Int,
    s2::Int,
    s3::Int,
    out::CuDeviceArray{Float64},
    fun
)

    s1 = size(i_k_1)[2]
    s2 = length(s_k)
    s3 = size(j_k)[2]
    
    if i <= s1 * s2 * s3
        first = i % (s1 * s2 * s3) ÷ (s1 * s2)
        second = i % (s1 * s2 * s3) % (s1 * s2) ÷ s1
        third = i % (s1 * s2 * s3) % (s1 * s2) % s1

        vec = vcat(i_k_1[:, first], s_k[second], j_k[:, third])

        out[first, second, third] = func(vec)
    end
    return nothing
end

function gpu_test_kernel(
    i_k_1::CuArray{Float64},
    s_k::CuArray{Float64},
    j_k::CuArray{Float64},
    fun
)
    s1 = size(i_k_1)[2]
    s2 = length(s_k)
    s3 = size(j_k)[2]

    output = CUDA.zeros(Float64, (s1, s2, s3))

    kern = @cuda launch=false os_kernel(i_k_1, s_k, j_k, )


end


"""
int main()
{
    int x = 4;
    int y = 4;
    int z = 4;

    for (int i = 0; i < 100; i++) {
        int fourth_index = i / (x * y * z);
        int third_index = i % (x * y * z) / (x * y);
        int second_index = i % (x * y * z) % (x * y) / x;
        int first_index = i % (x * y * z) % (x * y) % x;

        printf("%d: (%d, %d, %d, %d)\n", i, first_index, second_index, third_index, fourth_index);
    }
}

int main()
{
    int x = 4;
    int y = 4;
    int z = 4;

    for (int i = 0; i < 100; i++) {
        int first_index = i % x;
        int second_index = i / x % y;
        int third_index = i / x / y % z;
        int fourth_index = i / x / y / z;

        printf("%d: (%d, %d, %d, %d)\n", i, first_index, second_index, third_index, fourth_index);
    }
}

"""

# This seems to be the goal -> Just need to check dimensions

let 
    x = 3
    y = 2
    z = 3
    for i in 0:17
        # first = i % x
        # second = i / x % y
        # third = i / x / y % z
        # fourth = i / x / y / z

        first = i % (x * y * z) ÷ (x * y)
        second = i % (x * y * z) % (x * y) ÷ x
        third = i % (x * y * z) % (x * y) % x

        println([i, first + 1, second + 1, third + 1])
    end
end

