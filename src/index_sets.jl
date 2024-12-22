# TODO Check if the greedy search can introduce duplicated indices when degeneracy is at play

function compute_left_index_sets(currentI::AbstractMatrix, Il::AbstractMatrix, sl::Vector)
    n_cols = size(Il, 2) * length(s)
    n_rows = size(Il, 1) + 1
    Il_t_sl = Matrix{Float64}(undef, n_rows, n_cols)

    currentI_pos = []
    col_index = 1
    for coll in eachcol(Il)
        for s in sl
            column = vcat(coll, s)
            Il_t_sl[:, col_index] = column
            if any(column .== eachcol(currentI))
                append!(currentI_pos, col_index)
            end
            col_index += 1
        end
    end
    return Il_t_sl, currentI_pos
end

function compute_right_index_sets(currentJ::AbstractMatrix, Jr::AbstractMatrix, sr::Vector)
    n_cols = size(Jr, 2) * length(sr)
    n_rows = size(Jr, 1) + 1
    sr_t_Jr = Matrix{Float64}(undef, n_rows, n_cols)

    currentJ_pos = []
    col_index = 1
    for s in sr
        for colJ in eachcol(Jr)
            column = vcat(s, colJ)
            sr_t_Jr[:, col_index] = column
            if any(column .== eachcol(currentJ))
                append!(currentI_pos, col_index)
            end
            col_index += 1
        end
    end
    return sr_t_Jr, currentJ_pos
end

function compute_index_sets(
    currentI::AbstractMatrix, 
    currentJ::AbstractMatrix,
    Il::AbstractMatrix, 
    sl::Vector, 
    sr::Vector, 
    Jr::AbstractMatrix
)
    Il_t_sl, currentI_pos = compute_left_index_sets(currentI, Il, sl)
    sr_t_Jr, currentJ_pos = compute_right_index_sets(currentJ, Jr, sr)

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos
end

function compute_index_sets(
    currentI::AbstractMatrix,
    currentJ::AbstractMatrix,
    sl::Vector,
    sr::Vector, 
    Jr::AbstractMatrix
)
    sr_t_Jr, currentJ_pos = compute_right_index_sets(currentJ, Jr, sr)

    currentI_vec = reshape(CurrentI, (size(currentI)[2]))
    currentI_pos = [findfirst(isequal(x), currentI_vec) for x in sl]

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos   
end

function compute_index_sets(
    currentI::AbstractMatrix, 
    currentJ::AbstractMatrix,
    Il::AbstractMatrix, 
    sl::Vector, 
    sr::Vector
)
    Il_t_sl, currentI_pos = compute_left_index_sets(currentI, Il, sl)

    currentJ_vec = reshape(currentJ, (size(currentJ)[2]))
    currentJ_pos = [findfirst(isequal(x), currentJ_vec) for x in sr]

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos   
end


