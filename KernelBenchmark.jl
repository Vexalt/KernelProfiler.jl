module KernelBenchmark

using CUDA

include("KernelUtils.jl")
include("KernelPlot.jl")

using .KernelUtils: create_buffer, global_index, set_threads, THREADS, BLOCKS, global_index
using .KernelPlot: plot_benchmark

"""
    benchmarked_instructions

An array of benchmarked kernel instructions.
"""
global benchmarked_instructions = []

export @debug_kernel, benchmark_function, set_threads, plot_benchmark

"""
Insert timing to all expressions in a given expression for profiling.

# Arguments
- `expr::Expr`: the expression to insert timing into.
- `index`: keeps track of the current expression's index.
- `instructions`: a reference to a list of instructions (expressions).

# Returns
- `new_args`: a new expression with timing inserted into each expression in the original expression.
"""
function insert_timing(expr::Expr, index, instructions)
    @info "Processing expression: ", expr
    new_args = Expr(:block)

    for arg in expr.args
        if arg isa Expr
            if arg.head in (:macrocall, :call, :(=), :(+=), :(:-=), :(:*=), :(:/=), :(:%), :(^), 
                             :(:<), :(:<=), :(==), :(:!=), :(:>=), :(:>), :(:&&), :(:||), :(:!), 
                             :(:&), :(:|), :(:>>), :(:>>>), :(:<<))

                start_time_var = Symbol("start_", :($(index[])))
                end_time_var = Symbol("end_", :($(index[])))

                # Add timing start
                push!(new_args.args, :($(start_time_var) = clock(UInt64)))

                # Add original expression
                push!(new_args.args, arg)

                # Add timing end
                push!(new_args.args, :($(end_time_var) = clock(UInt64)))

                # Add timing calculation
                push!(new_args.args, :(gpu_arr[id+$(index[]*THREADS[])] = $(end_time_var)-$(start_time_var)))

                # Increment expression index
                index[] += 1

                # Store original expression in instructions
                push!(instructions[], arg)

                @warn "Processed expression: ", arg
            else
                # Recurse into nested expressions (blocks, conditionals, loops)
                if arg.head in (:block, :if, :for, :while)
                    # Insert timing into nested expression
                    arg.args = insert_timing(arg, index, instructions)

                    # Add processed expression to new_args
                    push!(new_args.args, arg)
                else
                    @error "Unexpected expression type: ", arg.head
                    push!(new_args.args, arg)
                end
            end
        else
            #push!(new_args.args, arg)
        end
    end

    return new_args
end

"""
    debug_kernel(func_expr::Expr)

A macro that modifies a function to benchmark each instruction executed in the GPU kernel.

# Arguments
- `func_expr::Expr`: A Julia expression that represents a function to be benchmarked.

# Returns
- `Expr`: A new Julia expression that represents the modified function.
"""
macro debug_kernel(func_expr::Expr)
    if !(isa(func_expr, Expr) && func_expr.head == :function)
        error("The @debug_kernel macro should be applied to a function definition.")
    end
    
    func_name = func_expr.args[1].args[1]
    func_args = func_expr.args[1].args[2:end]
    pushfirst!(func_expr.args[2].args, :(id = threadIdx().x + (blockIdx().x - 1) * blockDim().x))

    benchmark_instructions = Ref([])
    global_index[] = 0
    func_body = insert_timing(func_expr.args[2], global_index, benchmark_instructions)
    @info "TOTAL SIZE: $(global_index[])"
    global benchmarked_instructions = benchmark_instructions[]

    new_func = Expr(:function, Expr(:call, func_name, func_args..., :gpu_arr), func_body)
    @info new_func
    return new_func
end

"""
    benchmark_function(func, args...)

Benchmark a given function on the GPU.

# Arguments
- `func`: A function to be benchmarked.
"""
function benchmark_function(func, args...)
    cpu_arr, gpu_arr = create_buffer(global_index[])

    @cuda blocks=BLOCKS[] threads=THREADS[] func(args..., gpu_arr)

    CUDA.synchronize()

    # Return benchmark results
    plot_benchmark(cpu_arr, THREADS[], global_index[], benchmarked_instructions)
end

end # module
