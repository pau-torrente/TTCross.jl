using Test
include("../src/utils.jl")

# TODO Adapt tests to Matrix structure instead of nested vectors.
@testset "InterpolatingCrosses" begin
    @testset "constructor with n_var" begin
        ic = InterpolatingCrosses(3)
        @test ic.n_var == 3
        @testset "cross sizes" begin
            for k in 1:ic.n_var-1
                @test isa(ic.calI[k], AbstractMatrix) && size(ic.calI[k]) == (k, 0)
                @test isa(ic.calJ[k], AbstractMatrix) && size(ic.calJ[k]) == (ic.n_var-k, 0)
            end
        end
        @test size(ic.calI)[1] == 2
        @test size(ic.calJ)[1] == 2
        @test ic.cross_sizes == [0, 0]
        
    end

    # TODO Account for actual meaning of I and J (extremal cases, sizes, nestedness, etc.)
    @testset "constructor with calI and calJ" begin
        calI = [[1.  2.;], 
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
                 [3. 5.;]
        ]

        ic = InterpolatingCrosses(calI, calJ)

        @test ic.calI == [[1.  2.;], 
                        [1. 1. 2.; 
                        1. 2. 2.], 
                        [1. 1.;
                        2. 2.;
                        3. 4.]
                    ]
        @test ic.calJ == [[1.  1.;
                        2.  2.;
                        3.  4.], 
                        [2.  2.  2.;
                        3.  5.  4.],
                        [3. 5.;]
                    ]
        @test ic.cross_sizes == [2, 3, 2]
        @test ic.n_var == 4
    end

    @testset "changing the index sets" begin
        ic = InterpolatingCrosses(3)

        newI = reshape([1])
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

