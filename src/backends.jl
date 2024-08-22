const _use_cuda = Ref(false)

function use_cuda(flag::Bool)
    _use_cuda[] = flag
    if flag
        using CUDA
        println("CUDA backend acivated.")
    else
        println("CUDA backend deactivated. Now using CPU.")
    end
end
