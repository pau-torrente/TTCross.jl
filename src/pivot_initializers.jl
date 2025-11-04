#TODO Refactor this into matrices to fit with the rest of the framework

function random_init_sets(grid::Grid, set_sizes::Int)
    size_list = [set_sizes for _ in 1:Grid.n_var]
    return random_init_sets(grid, size_list)
end

function random_init_sets(grid::Grid, set_sizes::Vector{Int})
    
end


function firstn_init_sets(grid::Grid, set_sizes::Int)
    size_list = [set_sizes for _ in 1:Grid.n_var]
    return random_init_sets(grid, size_list)
end

function firstn_init_sets(grid::Grid, set_sizes::Vector{Int})

end

^