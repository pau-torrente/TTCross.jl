using Test

@testset "InterpolatingCrosses" begin
    @testset "constructor with n_var" begin
        using ..Interpolators
        ic = InterpolatingCrosses(3)
        @test ic.calI == [[], [], []]
        @test ic.calJ == [[], [], []]
        @test ic.cross_sizes == [0, 0, 0]
        @test ic.n_var == 3
    end

    # TODO Account for actual meaning of I and J (extremal cases, sizes, nestedness, etc.)
    @testset "constructor with calI and calJ" begin
        using ..Interpolators
        calI = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        calJ = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        ic = InterpolatingCrosses(calI, calJ)
        @test ic.calI == calI
        @test ic.calJ == calJ
        @test ic.cross_sizes == [2, 3, 4]
        @test ic.n_var == 3
    end
end