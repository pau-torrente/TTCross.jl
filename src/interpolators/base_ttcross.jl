module BaseTTCross

export InterpolatingCrosses, Grid

mutable struct InterpolatingCrosses
    calI::Vector{Vector{Int}}
    calJ::Vector{Vector{Int}}
    cross_sizes::Vector{Int}
    n_var::Int

    # TODO Check if calI and calJ should have another nested dimension
    function InterpolatingCrosses(n_var::Int)
        calI = Vector{Vector{Int}}(undef, n_var)
        calJ = Vector{Vector{Int}}(undef, n_var)
        cross_sizes = Vector{Int}(undef, n_var)
        for k in 1:n_var
            calI[k] = Vector{Int}()
            calJ[k] = Vector{Int}()
        end
        new(calI, calJ, cross_sizes, n_var)
    end

    function InterpolatingCrosses(calI::Vector{Vector{Int}}, calJ::Vector{Vector{Int}})
        if length(calI) != length(calJ)
            throw(ArgumentError("calI and calJ must have the same length"))
        end
        n_var = length(calI)
        cross_sizes = Vector{Int}(undef, n)
        for k in 1:n
            if length(calI[k]) != length(calJ[k])
                throw(ArgumentError("calI and calJ must have the same length, but got mismatch at index $k"))
            end
            cross_sizes[k] = length(calI[k])
        end
        new(calI, calJ, cross_sizes, n_var)
    end
end

struct Grid
    intervals::Vector{Vector{Float64}}
    grid_points::Vector{Vector{Float64}}
    points_per_var::Vector{Int}
    n_var::Int

    function Grid(intervals::Vector{Vector{Float64}}, points_per_var::Vector{Int})
        if length(intervals) != length(points_per_var)
            throw(ArgumentError("intervals and points_per_var must have the same length"))
        end
        n_var = length(intervals)
        grid_points = Vector{Vector{Float64}}(undef, n_var)
        for k in 1:n_var
            grid_points[k] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var[k]))
        end
        new(intervals, grid_points, points_per_var, n_var)
    end

    function Grid(grid_points::Vector{Vector{Float64}})
        n_var = length(grid_points)
        intervals = Vector{Vector{Float64}}(undef, n_var)
        points_per_var = Vector{Int}(undef, n_var)
        for k in 1:n_var
            intervals[k] = [grid_points[k][1], grid_points[k][end]]
            points_per_var[k] = length(grid_points[k])
        end
        new(intervals, grid_points, points_per_var, n_var)
    end
end
end

