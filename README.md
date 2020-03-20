# cpu-gpu

This little program launches an identical workload on both CPU and GPU and see how "complicated" the workload has to be for GPU to beat CPU, in terms of covering the memory copy overhead and competing with CPU simd.

The workload is simulated by `n` instructions operating on `size` operands (number, not bytes), where:
* a single instruction is a multiply by the operand itself
* `n` incresments from 0 to 128, can be modified in code
* `size` could be specified as the first argument of this program, defaults to `256 * 1024 * 1024` (again, number, not bytes)


