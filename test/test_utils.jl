using Test
include("../src/utils.jl")

# TODO Adapt tests to Matrix structure instead of nested vectors.
@testset "InterpolatingCrosses" begin
    @testset "constructor with n_var" begin
        ic = InterpolatingCrosses(3)
        @test all(x -> isa(x, AbstractMatrix) && size(x) == (0, 0), ic.calI)
        @test all(x -> isa(x, AbstractMatrix) && size(x) == (0, 0), ic.calJ)
        @test size(ic.calI)[1] == 2
        @test size(ic.calJ)[1] == 2
        @test ic.cross_sizes == [0, 0]
        @test ic.n_var == 3
    end

    # TODO Account for actual meaning of I and J (extremal cases, sizes, nestedness, etc.)
    @testset "constructor with calI and calJ" begin
        calI = [[1. ; 2.], 
                [1. 1. 2.; 
                 1. 2. 2.], 
                [1. 1.;
                 2. 2.;
                 3. 4.]
        ]
        
        calJ = [[1.  1.;
                 2.  2.;
                 3.  4.], 
                 [2.  2.  2.;
                 3.  5.  4.],
                 [3.; 5.]
        ]

        ic = InterpolatingCrosses(calI, calJ)
        @test ic.calI == [reshape([1. ; 2.], (1,2)), transpose([1. 1. ; 1. 2.; 2. 2.]), transpose([1. 2. 3.; 1. 2. 4.])]
        @test ic.calJ == [transpose([1. 2. 3.; 1. 2. 4.]), transpose([2. 3.; 2. 5.; 2. 4.]), reshape([3.; 5.], (1, 2))]
        @test ic.cross_sizes == [2, 3, 2]
        @test ic.n_var == 4
    end
end


# TODO Decide whether we should differentiate square grids from non square ones. -> Array indexing seems more performant
@testset "Grid" begin
    @testset "constructor with intervals and points_per_var" begin
        intervals = [[0., 1.], [1., 2.], [2., 3.]]
        points_per_var = [2, 2, 2]
        g = Grid(intervals, points_per_var)
        @test g.intervals == intervals
        @test g.points == [[0., 1.], [1., 2.], [2., 3.]]
        @test g.points_per_var == points_per_var
        @test g.n_var == 3
    end

    @testset "constructor with intervals and integer points_per_var" begin
        intervals = [[0., 1.], [1., 2.], [2., 3.]]
        points_per_var = 2
        g = Grid(intervals, points_per_var)
        @test g.intervals == intervals
        @test g.points == [[0., 1.], [1., 2.], [2., 3.]]
        @test g.n_var == 3
        @test g.points_per_var == [points_per_var for _ in 1:3]
        
    end
    @testset "constructor with points" begin
        points = [[0., 1.], [1., 1.5, 2.], [2., 2.5, 3., 3.5]]
        g = Grid(points)
        @test g.intervals == [[0., 1.], [1., 2.], [2., 3.5]]
        @test g.points == points
        @test g.points_per_var == [2, 3, 4]
        @test g.n_var == 3
    end
end

