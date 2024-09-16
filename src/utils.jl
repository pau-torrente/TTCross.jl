mutable struct InterpolatingCrosses
    calI::Vector{Matrix{Float64}}
    calJ::Vector{Matrix{Float64}}
    cross_sizes::Vector{Int}
    n_var::Int

    # TODO Check if calI and calJ should have another nested dimension
    function InterpolatingCrosses(n_var::Int)
        calI = Vector{Matrix{Float64}}(undef, n_var)
        calJ = Vector{Matrix{Float64}}(undef, n_var)
        cross_sizes = Vector{Int}(undef, n_var)
        calI[1] = Matrix{Float64}(undef, 0, 0)
        calJ[end] = Matrix{Float64}(undef, 0, 0)

        for k in 1:n_var-1
            calI[k+1] = Matrix{Float64}(undef, 0, 0)
            calJ[k] = Matrix{Float64}(undef, 0, 0)
        end
        new(calI, calJ, cross_sizes, n_var)
    end

    function InterpolatingCrosses(calI::Vector{Vector{Vector{Float64}}}, calJ::Vector{Vector{Vector{Float64}}})
        if length(calI) != length(calJ)
            throw(ArgumentError("calI and calJ must have the same length"))
        end
        n_var = length(calI)
        cross_sizes = Vector{Int}(undef, n)
        for k in 1:n
            if size(calI[k])[1] != size(calJ[k])[1]
                throw(ArgumentError("calI and calJ must have the same length, but got mismatch at index $k"))
            end
            cross_sizes[k] = size(calI[k])[1]
        end
        new(calI, calJ, cross_sizes, n_var)
    end
end

struct Grid
    intervals::Matrix{Float64}
    points::Matrix{Float64}
    points_per_var::Vector{Int}
    n_var::Int

    function Grid(intervals::Matrix{Float64}, points_per_var::Vector{Int})
        if size(intervals)[1] != length(points_per_var)
            throw(ArgumentError("intervals and points_per_var must have the same length"))
        end
        n_var = length(intervals)
        points = Matrix{Float64}(undef, n_var)
        for k in 1:n_var
            points[k, :] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var[k]))
        end
        new(intervals, points, points_per_var, n_var)
    end

    function Grid(points::Matrix{Float64})
        n_var = length(points)
        intervals = Matrix{Float64}(undef, n_var)
        points_per_var = Vector{Int}(undef, n_var)
        for k in 1:n_var
            intervals[k] = [points[k][1], points[k][end]]
            points_per_var[k] = length(points[k])
        end
        new(intervals, points, points_per_var, n_var)
    end
end

global _backend::Symbol = :cpu

function set_backend(new_backend::Symbol)
    if new_backend == :cpu
        global _backend = :cpu
        println("Backend set to CPU")
    elseif new_backend == :gpu
        global _backend = :gpu
        println("Backend set to GPU")
    else
        DomainError("Invalid backend")
    end        
end

