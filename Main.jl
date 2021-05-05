include("./ParticlesSwarmOptimization.jl")
# using .ParticlesSwarmOptimization


options = Dict("size" => 50, 
                "dim" => 1,
                "max_iterations" => 100,
                "frac_elite"=> 0.025,
                "selection_data" => "stud",
                "crossover_data"=> 0.1,
                "mutation_rate"=> 0.05,
                "mutation_data"=>0.1,
                "lower_bound" => -5,
                "upper_bound" => 5
                )

                
function  sumsquare(x)
    return sum(x.^2)    
end

function  loss(β, x, y)
    return sum( (β*x - y).^2)    
end


PSO_optimize(sumsquare, options)
