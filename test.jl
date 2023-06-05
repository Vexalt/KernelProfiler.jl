include("KernelBenchmark.jl")
using .KernelBenchmark

set_threads(10)

kernel = @debug_kernel function kernel(val)
    @cuprintln val
    return
end

benchmark_function(kernel, 0)