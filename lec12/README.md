- cache_01_slow.c: The program does not perform well because the matrix is read by column rather than by row
- cache_01_fast.c: Same as cache_01_slow.c, but the matrix is read by row rather than by column, thus outperforming the previous version
- cache_01_fast_broken.c: Same as cache_01_fast.c, but the vector declaration is within the main body. This enables GCC to apply Dead Code Elimination (DCE), and remove basically all the code. The application would then just not compute anything. Beware, some compilers can apply dead code elimination also to global variable, and in that case the cache_01_slow.c and cache_01_fast.c would also not compute anything.
- cache_01_fast_broken_fixed.c Same as cache_01_fast_broken_fixed.c, but now we do something with the result of the calculation (e.g., print the sum of the elements of y), so that the compiler does not eliminate the code. Alternatively, you can remove DCE by adding the following flags when compiling: -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse


For more examples, check https://github.com/Kobzol/hardware-effects

To install perf on WSL2:
```bash
sudo apt install linux-tools-generic
```