# cpu-gpu

This little program launches an identical workload on both CPU and GPU and see how "complicated" the workload has to be for GPU to beat CPU, in terms of covering the memory copy overhead and competing with CPU simd.

The workload is simulated by `n` instructions operating on `size` operands, where:
* `n` incresments from 0 to 128, can be modified in code
* a single instruction is a multiply by the operand itself
* `size` could be specified as the first argument of this program, defaults to `256 * 1024 * 1024`


