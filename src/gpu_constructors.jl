using CUDA

function _gpu_2site_block(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Vector{Vector{Float64}},
    func::Function,
)

end

function _gpu_2site_block(
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    j_kp1::Vector{Vector{Float64}},
    func::Function,
)

end

function _gpu_2site_block(
    i_k_1::Vector{Vector{Float64}}, 
    s_k::Vector{Float64}, 
    s_kp1::Vector{Float64}, 
    func::Function,
)

end

function _gpu_1site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    j_k::Vector{Float64},
    func::Function,
)

end

function _gpu_1site_block(
    i_k_1::Vector{Vector{Float64}},
    s_k::Vector{Float64},
    func::Function,
)

end

function _gpu_1site_block(
    s_k::Vector{Float64},
    j_k::Vector{Float64},
    func::Function,
)

end

function _gpu_inverse_block(
    i_k::Vector{Vector{Float64}},
    j_k::Vector{Vector{Float64}},
    func::Function,
)

end