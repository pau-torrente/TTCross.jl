mutable struct InterpolatingCrosses
    calI::Vector{AbstractMatrix}
    calJ::Vector{AbstractMatrix}
    cross_sizes::Vector{Int}
    n_var::Int

    # TODO Check if calI and calJ should have another nested dimension
    function InterpolatingCrosses(n_var::Int)
        calI = Vector{AbstractMatrix}(undef, n_var-1)
        calJ = Vector{AbstractMatrix}(undef, n_var-1)
        cross_sizes = [0 for _ in 1:n_var-1]

        for k in 1:n_var-1
            calI[k] = Matrix{AbstractMatrix}(undef, 0, 0)
            calJ[k] = Matrix{AbstractMatrix}(undef, 0, 0)
        end
        
        new(calI, calJ, cross_sizes, n_var)
    end

    # TODO I am forcing the input to float64, which is probably desired from a numerical point of view, but not very flexible
    # I am also very unsure about typing things as <:Type
    function InterpolatingCrosses(calI::Vector{<:AbstractArray{Float64}}, calJ::Vector{<:AbstractArray{Float64}})
        if size(calI)[1] != size(calJ)[1]
            throw(ArgumentError("calI and calJ must have the same length"))
        end
        processing_calI = copy(calI)
        processing_calJ = copy(calJ)

        n_var = size(processing_calI)[1] + 1
        cross_sizes = Vector{Int}(undef, n_var-1)

        if typeof(processing_calI[1]) <: Vector
            processing_calI[1] = reshape(processing_calI[1], (1, size(processing_calI[1])[1]))
        end

        if typeof(processing_calJ[end]) <: Vector
            processing_calJ[end] = reshape(processing_calJ[end], (1, size(processing_calJ[end])[1]))
        end

        for k in 1:n_var-1
            if length(size(processing_calI[k])) != 2 && length(size(processing_calI[k])) != 2
                throw(ArgumentError("{I_1,...,I_N-1} and {J_1,...,J_N-1} should be vectors of arrays."))
            end

            if size(processing_calI[k])[2] != size(processing_calJ[k])[2]
                throw(ArgumentError("calI and calJ must have the same length, but got mismatch at index $k"))   
            end    

            cross_sizes[k] = size(processing_calI[k])[2]
        end
        new(processing_calI, processing_calJ, cross_sizes, n_var)
    end
end

struct Grid
    intervals::Vector{Vector{Float64}}
    points::Vector{Vector{Float64}}
    points_per_var::Vector{Int}
    n_var::Int

    function Grid(intervals::Vector{Vector{Float64}}, points_per_var::Vector{Int})
        n_var = length(intervals)
        points = Vector{Vector{Float64}}(undef, n_var)
        for k in 1:n_var
            points[k] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var[k]))
        end
        new(intervals, points, points_per_var, n_var)
    end

    function Grid(intervals::Vector{Vector{Float64}}, points_per_var::Int)
        n_var = length(intervals)
        points = Vector{Vector{Float64}}(undef, n_var)
        points_per_var_list = [points_per_var for _ in 1:n_var]
        for k in 1:n_var
            points[k] = collect(range(intervals[k][1], stop=intervals[k][2], length=points_per_var_list[k]))
        end
        new(intervals, points, points_per_var_list, n_var)
    end

    function Grid(points::Vector{Vector{Float64}})
        n_var = length(points)
        intervals = Vector{Vector{Float64}}(undef, n_var)
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

function check_backend()
    return _backend
end

