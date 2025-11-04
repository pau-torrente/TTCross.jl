struct Grid{T<:AbstractFloat}
    intervals::Vector{Vector{T}}
    points::Vector{Vector{T}}
    points_per_var::Vector{Int}
    n_var::Int

    function Grid(intervals::Vector{Vector{T}}, points_per_var::Vector{Int}) where T<:AbstractFloat
        n_var = length(intervals)
        points = Vector{Vector{T}}(undef, n_var)
        for k in 1:n_var
            points[k] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var[k]))
        end
        new{T}(intervals, points, points_per_var, n_var)
    end

    function Grid(intervals::Vector{Vector{T}}, points_per_var::Int) where T<:AbstractFloat
        n_var = length(intervals)
        points = Vector{Vector{T}}(undef, n_var)
        points_per_var_list = [points_per_var for _ in 1:n_var]
        for k in 1:n_var
            points[k] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var_list[k]))
        end
        new{T}(intervals, points, points_per_var_list, n_var)
    end

    function Grid(points::Vector{Vector{T}}) where T<:AbstractFloat
        n_var = length(points)
        intervals = Vector{Vector{T}}(undef, n_var)
        points_per_var = Vector{Int}(undef, n_var)
        for k in 1:n_var
            intervals[k] = [points[k][1], points[k][end]]
            points_per_var[k] = length(points[k])
        end
        new{T}(intervals, points, points_per_var, n_var)
    end
end

abstract type AbstractBackend end

struct CpuBackend <: AbstractBackend end
struct GpuBackend <: AbstractBackend end

const CURRENT_BACKEND = Ref{AbstractBackend}(CpuBackend())

function set_backend(backend::AbstractBackend)
    CURRENT_BACKEND[] = backend
end
