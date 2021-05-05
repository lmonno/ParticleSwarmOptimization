

using Distributed
addprocs(2, exeflags="--project=.")
@everywhere begin
    using Distributed
    using StatsBase
    using BenchmarkTools
end
data = rand(10_000,20_000)    
@everywhere function t2(d1,d2)
    append!(d1,d2)
    d1
end
@btime begin 
    res =  @distributed (t2) for col = 1:size(data)[2]
        [(myid(),col, StatsBase.mean(data[:,col]))]
    end
end


addprocs(2, exeflags="--project=.")
nworkers()
@everywhere begin
    using Distributed
    using StatsBase
    using BenchmarkTools
end

data = rand(10_000,20_000)  
@everywhere function t2(d1,d2)
    append!(d1,d2)
    d1
end
@btime begin 
    res =  @distributed (t2) for col = 1:size(data)[2]
        [(myid(),col, StatsBase.mean(data[:,col]))]
    end
end

wa


# using Distributed
# #https://stackoverflow.com/questions/52432639/distributed-seems-to-work-function-return-is-wonky

# rmprocs(nworkers()); addprocs(4); nworkers()


# @everywhere using Statistics

# function setup(n, B, nl, sl)
#     @distributed (+) for m in 1:nl
#         Y = randn(n)
#         mean_Y = mean(Y)
#         counter = 0
#         for i = 1:B
#                 Y_st = rand(Y, n)
#                 mean_Y_st = mean(Y_st)  
#                 if mean_Y_st > mean_Y
#                     counter += 1
#                 end
#         end
#         p_value = counter / B
#         Int(p_value < sl)
#     end
# end

# take_counter = setup(25, 200, 40, 0.05)
# ####