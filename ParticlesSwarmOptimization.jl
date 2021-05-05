#push!(LOAD_PATH, pwd()*"\\src")
#module ParticleSwarmOptimization
using StatsBase, Distributed

mutable struct PSO
    population::Matrix{Float64}
    options::Dict
    iter::Int64
    rankind
    child::Matrix{Float64}
    num_child::Int64 
    parents::Matrix{Int64} # size x 2, 
                             # per riga indice dei parents, 
                             # per colonna i due genitori
    num_par::Int64 
    elite::Vector{Int64} # size , 
    num_elite::Int64 
    fval::Vector{Float64}
    num_feval::Int64 
    f_best::Float64 
    x_best::Vector{Float64}
    obj_func
    PSO() = new()
end


function set_options(aPSO::PSO, options::Dict)
    aPSO.options = options    
end

function PSO_optimize(obj_func, options::Dict)
    myPSO = PSO()
    set_options(myPSO, options)
    myPSO.obj_func = obj_func
    PSO_inizialize(myPSO)
    while myPSO.iter < myPSO.options["max_iterations"]
        PSO_make_step(myPSO)
        @info myPSO.iter myPSO.f_best myPSO.x_best 
    end  
    return myPSO.x_best 
end

function PSO_make_step(aPSO::PSO)
    aPSO.iter += 1
    PSO_choose_parents(aPSO)    
    PSO_apply_crossover(aPSO)    
    # self.apply_mutation()
    PSO_add_children_to_population(aPSO)  
    PSO_eval_fitness(aPSO)  
    PSO_do_ranking(aPSO)  
end

function PSO_inizialize_polulation(aPSO::PSO)
    lb = aPSO.options["lower_bound"]
    ub = aPSO.options["upper_bound"]
    size = aPSO.options["size"]
    dim = aPSO.options["dim"]
    aPSO.population = lb .+ (ub-lb) .* rand(Float64, (size, dim))
    aPSO.rankind = 1:size |> collect
    aPSO.fval = zeros(size)
    aPSO.child = zeros((2*aPSO.num_par, dim))
    aPSO.parents = zeros(Int64,(aPSO.num_par, 2))
    aPSO.elite = zeros(Int64,aPSO.num_elite)
    aPSO.f_best = 0.0

end


function PSO_inizialize(aPSO::PSO)
    aPSO.iter = 0
    aPSO.num_feval = 0
    aPSO.num_elite = (round(Int64, aPSO.options["frac_elite"] * aPSO.options["size"]))
    
    aPSO.num_child = aPSO.options["size"] - aPSO.num_elite

    
    if aPSO.num_child % 2 != 0
        aPSO.num_child -= 1
        aPSO.num_elite += 1 
    end

    aPSO.num_par = aPSO.num_child ÷ 2
    PSO_inizialize_polulation(aPSO)
    PSO_eval_fitness(aPSO)
    PSO_do_ranking(aPSO)
end

function PSO_eval_fitness(aPSO::PSO)
    # da parallelizzare
    aPSO.fval = mapslices(aPSO.obj_func, aPSO.population,  dims = 2) |> vec
    
end


function PSO_choose_parents(aPSO::PSO)    
    
    fitness = (aPSO.options["size"]:-1:1 |> collect) ./ 
                (aPSO.options["size"]*(aPSO.options["size"]+1)/2)
    # fiteness bassa
    if aPSO.options["selection_data"] == "stud"
        aPSO.parents[:,1] = fill(aPSO.rankind[1], (1,aPSO.num_par)) 
    else
        selected = sample(1:aPSO.options["size"], Weights(fitness), aPSO.num_par; replace = true)
        aPSO.parents[:,1] = aPSO.rankind[selected]
    end
    selected = sample(1:aPSO.options["size"] , Weights(fitness), aPSO.num_par; replace = true)
    aPSO.parents[:,2] = aPSO.rankind[selected ] 
end

function PSO_apply_crossover(aPSO::PSO)
    x_min = min.(aPSO.population[aPSO.parents[:,1],:],
                aPSO.population[aPSO.parents[:,2],:])
    x_max = max.(aPSO.population[aPSO.parents[:,1],:],
                aPSO.population[aPSO.parents[:,2],:])
    δ = aPSO.options["crossover_data"] * (x_max-x_min)
    lw = max.(x_min - δ, aPSO.options["lower_bound"])
    hg = min.(x_max + δ, aPSO.options["upper_bound"])
    u = rand(Float64, (aPSO.num_par,aPSO.options["dim"]))
    urca = (u .* lw) .+ ((1 .- u) .* hg)
    aPSO.child[1:2:aPSO.num_child,:]=  u .* lw .+ (1 .- u) .* hg
    aPSO.child[2:2:aPSO.num_child,:]= (1 .- u) .* lw .+ u .* hg
end
function apply_mutation(aPSO::PSO)

end
function PSO_add_children_to_population(aPSO::PSO)
    all_index = 1:aPSO.options["size"]
    change_index = .!(all_index .∈ (aPSO.elite,))
    aPSO.population[change_index,:] = aPSO.child
end
function PSO_do_ranking(aPSO::PSO)
    aPSO.rankind = aPSO.fval |> vec |> sortperm
    aPSO.fval = aPSO.fval[aPSO.rankind]
    aPSO.f_best = aPSO.fval[1]
    aPSO.x_best = aPSO.population[aPSO.rankind[1],:]
    aPSO.elite = sort(aPSO.rankind[1:aPSO.num_elite])
end

#end



#@run PSO_optimize(sumsquare, options)


# a =  rand(Float64, (3, 10 ))
# fval = mapslices(sumsquare,a,  dims = 1)

# @run sumsquare(a)
