mutable struct InterpolatingCrosses{T<:AbstractFloat}
    calI::Vector{Matrix{T}}
    calJ::Vector{Matrix{T}}
    cross_sizes::Vector{Int}
    n_var::Int

    function InterpolatingCrosses(n_var::Int; T::Type{<:AbstractFloat}=Float64)
        calI = [Matrix{T}(undef, k, 0) for k in 1:n_var-1]
        calJ = [Matrix{T}(undef, n_var - k, 0) for k in 1:n_var-1]
        cross_sizes = zeros(Int, n_var - 1)
        return new{T}(calI, calJ, cross_sizes, n_var)
    end

    function InterpolatingCrosses(calI::Vector{Matrix{T}}, calJ::Vector{Matrix{T}}) where {T<:AbstractFloat}
        length(calI) == length(calJ) || throw(ArgumentError("calI and calJ must have same length"))
        n_var = length(calI) + 1
        cross_sizes = Int[size(calI[k], 2) for k in 1:length(calI)]
        return InterpolatingCrosses{T}(copy(calI), copy(calJ), cross_sizes, n_var)
    end
end

function add_calI!(crosses::InterpolatingCrosses, position::Int, new_cross::Matrix)
    @boundscheck 1 ≤ position ≤ crosses.n_var - 1 || throw(BoundsError(crosses.calI, position))
    old = crosses.calI[position]
    try
        crosses.calI[position] = hcat(old, new_cross)
        crosses.cross_sizes[position] = size(crosses.calI[position], 2)
    catch e
        if e isa DimensionMismatch
            throw(DimensionMismatch("New calI cross at position $position has wrong number of rows: expected $(size(old,1)), got $(size(new_cross,1))"))
        else
            rethrow(e)
        end
    end
    return crosses
end

function add_calJ!(crosses::InterpolatingCrosses, position::Int, new_cross::Matrix)
    @boundscheck 1 ≤ position ≤ crosses.n_var - 1 || throw(BoundsError(crosses.calJ, position))
    old = crosses.calJ[position]
    try
        crosses.calJ[position] = hcat(old, new_cross)
        crosses.cross_sizes[position] = size(crosses.calJ[position], 2)
    catch e
        if e isa DimensionMismatch
            throw(DimensionMismatch("New calJ cross at position $position has wrong number of rows: expected $(size(old,1)), got $(size(new_cross,1))"))
        else
            rethrow(e)
        end
    end
    return crosses
end

function add_calJ!(crosses::InterpolatingCrosses{T}, position::Int, new_cross::Matrix{T}) where {T<:AbstractFloat}
    @boundscheck 1 ≤ position ≤ crosses.n_var - 1 || throw(BoundsError(crosses.calJ, position))
    old = crosses.calJ[position]
    try
        crosses.calJ[position] = hcat(old, new_cross)
        crosses.cross_sizes[position] = size(crosses.calJ[position], 2)
    catch e
        if e isa DimensionMismatch
            throw(DimensionMismatch("New calJ cross at position $position has wrong number of rows: expected $(size(old,1)), got $(size(new_cross,1))"))
        else
            rethrow(e)
        end
    end
    return crosses
end

function add_crosses!(crosses::InterpolatingCrosses, position::Int, new_I::Matrix, new_J::Matrix)
    @boundscheck 1 ≤ position ≤ crosses.n_var - 1 || throw(BoundsError(crosses.calI, position))
    size(new_I, 2) == size(new_J, 2) ||
        throw(ArgumentError("Mismatched number of columns in new_I and new_J at position $position"))

    add_calI!(crosses, position, new_I)
    add_calJ!(crosses, position, new_J)
    return crosses
end

function find_index_set(full_set::Matrix{T}, new_set::Matrix{T}) where T<:AbstractFloat
    return [findfirst(full_col -> full_col == new_col, eachcol(full_set)) for new_col in eachcol(new_set)]
end

function compute_left_index_sets(currentI::Matrix{T}, Il::Matrix{T}, sl::Vector{T}) where T<:AbstractFloat
    Il_t_sl = vcat(repeat(Il, inner=(1, size(sl, 1))), reshape(repeat(sl, size(Il, 2)), 1, :))
    currentI_pos = find_index_set(Il_t_sl, currentI)
    return Il_t_sl, currentI_pos
end

function compute_right_index_sets(currentJ::Matrix{T}, Jr::Matrix{T}, sr::Vector{T}) where T<:AbstractFloat
    sr_t_Jr = vcat(reshape(repeat(sr, inner=(size(Jr, 2), 1)), 1, :), repeat(Jr, outer=(1, length(sr))))
    currentJ_pos = find_index_set(sr_t_Jr, currentJ)
    return sr_t_Jr, currentJ_pos
end

function compute_index_sets(
    currentI::Matrix{T},
    currentJ::Matrix{T},
    Il::Matrix{T},
    sl::Vector{T},
    sr::Vector{T},
    Jr::Matrix{T}
) where T<:AbstractFloat
    Il_t_sl, currentI_pos = compute_left_index_sets(currentI, Il, sl)
    sr_t_Jr, currentJ_pos = compute_right_index_sets(currentJ, Jr, sr)

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos
end

function compute_index_sets(
    currentI::Matrix{T},
    currentJ::Matrix{T},
    sl::Vector{T},
    sr::Vector{T},
    Jr::Matrix{T}
) where T<:AbstractFloat
    sr_t_Jr, currentJ_pos = compute_right_index_sets(currentJ, Jr, sr)

    currentI_vec = reshape(currentI, (size(currentI)[2]))
    currentI_pos = [findfirst(isequal(x), currentI_vec) for x in sl]

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos
end

function compute_index_sets(
    currentI::Matrix{T},
    currentJ::Matrix{T},
    Il::Matrix{T},
    sl::Vector{T},
    sr::Vector{T}
) where T<:AbstractFloat
    Il_t_sl, currentI_pos = compute_left_index_sets(currentI, Il, sl)

    currentJ_vec = reshape(currentJ, (size(currentJ)[2]))
    currentJ_pos = [findfirst(isequal(x), currentJ_vec) for x in sr]

    return Il_t_sl, sr_t_Jr, currentI_pos, currentJ_pos
end


