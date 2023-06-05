module KernelPlot

using PlotlyJS
using Colors
using Statistics
using CUDA

include("KernelUtils.jl")

using .KernelUtils: device_clock_rate

export plot_benchmark

"""
    rescale(arr, out_range)

Rescale the values of an array to fit within a given output range.

# Arguments
- `arr`: The array to be rescaled.
- `out_range`: The output range to rescale the array values to.

# Returns
- An array with rescaled values.
"""
function rescale(arr, out_range)
    min_val, max_val = 0, max(arr...)
    a, b = out_range[1], out_range[end]
    (b - a) .* ((arr .- min_val) ./ (max_val - min_val)) .+ a
end

"""
    plot_benchmark(cpu_arr, device_clock_rate)

Generate a plot for the kernel benchmark results.

# Arguments
- `cpu_arr`: The CPU array with the benchmark results.
- `device_clock_rate`: The clock rate of the device.

# Returns
- A PlotlyJS plot.
"""
function plot_benchmark(cpu_arr,threads, global_index, benchmarked_instructions)

    rate = (1 / device_clock_rate * 1000)

    color_range = range(colorant"rgb(50, 200, 50)", colorant"rgb(200,50,50)", length=99)
    matrix = reshape(cpu_arr, threads, global_index)
    means = mean(matrix, dims=1) .* rate
    mins = minimum(matrix, dims=1) .* rate
    maxs = maximum(matrix, dims=1) .* rate
    medians = median(matrix, dims=1) .* rate

    means_norm = round.(Int, rescale(means, 1:99))
    mins_norm = round.(Int, rescale(mins, 1:99))
    maxs_norm = round.(Int, rescale(maxs, 1:99))
    medians_norm = round.(Int, rescale(medians, 1:99))

    means_str = string.(round.(means; digits=5), " ms")
    mins_str = string.(round.(mins; digits=5), " ms")
    maxs_str = string.(round.(maxs; digits=5), " ms")
    medians_str = string.(round.(medians; digits=5), " ms")

    benchmarked_instructions = reshape(benchmarked_instructions, (1, global_index))
    data = [string.(benchmarked_instructions);means_str;mins_str;maxs_str;medians_str]
    data = permutedims(data)


    color_data = ["grey"; color_range[means_norm]; color_range[mins_norm]; color_range[maxs_norm]; color_range[medians_norm]]
    color_data = permutedims(color_data)

    plot(
        table(
            header=attr(
                values=["", "Means", "Mins", "Maxs", "Medians"],
                line_color="white", fill_color="gray",
                align="center", font=attr(color="black", size=12),
            ),
            cells=attr(
                values=data,
                line_color=color_data,
                fill_color=color_data,
                align="center", font=attr(color="white", size=11)
            )
        )
    )
end

end # module
