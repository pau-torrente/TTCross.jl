function maxvol_pivots(rect_matrix::AbstractMatrix, max_iter::Int, tol::Float64, full_indexset::Matrix{Float64})
    sz = size(rect_matrix)
    if sz[1] < sz[2]
        A = transpose(rect_matrix)
    else
        A = rect_matrix
    end

    maxvol_indices, _ = maxvol(rect_matrix, max_iter, tol)
    best_index_set = full_indexset[maxvol_indices]

    return maxvol_indices, best_index_set
end

function greedy_pivots(full_matrix::AbstractMatrix, currentI_pos::Vector{Int}, currentJ_pos::Vector{Int})
    core = full_matrix[currentI_pos, currentJ_pos]
    approx = full_matrix[:, currentJ_pos] * inv(core) * full_matrix[currentI_pos, :]
    max_value, max_index = findmax(approx)
    i_new = max_index[0]
    j_new = max_index[1]

    if i_new ∈ currentI_pos || j_new ∈ currentJ_pos || abs(full_matrix[i_new, j_new] - approx[i_new, j_new]) < tol
        return currentI_pos, currentJ_pos
    else
        newI_pos = append!(currentI_pos, i_new)
        newJ_pos = append!(currentJ_pos, j_new)
        return currentI_pos, currentJ_pos
    end
end