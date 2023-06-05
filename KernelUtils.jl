module KernelUtils

using CUDA

const device_clock_rate = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE)

"""
    THREADS

Number of threads used in the kernel.
"""
const THREADS = Ref(100)
const BLOCKS = Ref(1)

"""
    global_index

A reference to a global index that is used across the modules.
"""
const global_index = Ref(0)

export create_buffer, THREADS, global_index, set_threads

"""
    create_buffer(size::Int)

Create a buffer on the GPU and return both the CPU and GPU array representations.

# Arguments
- `size::Int`: The size of the buffer to create.

# Returns
- `Tuple`: A tuple with the CPU array as the first element and the GPU array as the second element.
"""
function create_buffer(size::Int)
    host_buf = CUDA.Mem.alloc(CUDA.Mem.HostBuffer, sizeof(Int) * size * THREADS[])
    gpu_ptr = convert(CuPtr{Int}, host_buf)
    cpu_ptr = convert(Ptr{Int}, host_buf)
    
    cpu_arr = unsafe_wrap(Array, cpu_ptr, size * THREADS[])
    gpu_arr = unsafe_wrap(CuArray, gpu_ptr, size * THREADS[])
    
    CUDA.synchronize()
    
    return cpu_arr, gpu_arr
end

"""
    set_threads(num::Int)

Set the number of threads used in the kernel.

# Arguments
- `num::Int`: The number of threads to set.
"""
function set_threads(num::Int)
    THREADS[] = max(1024, num)
    BLOCKS[] = ceil(num / 1024)
end

end # module
