- cache_01_slow.c: The program does not perform well because the matrix is read by column rather than by row
- cache_01_fast.c: Same as cache_01_slow.c, but the matrix is read by row rather than by column, thus outperforming the previous version
- cache_01_fast_broken.c: Same as cache_01_fast.c, but the vector declaration is within the main body. This enables GCC to apply Dead Code Elimination (DCE), and remove basically all the code. The application would then just not compute anything. Beware, some compilers can apply dead code elimination also to global variable, and in that case the cache_01_slow.c and cache_01_fast.c would also not compute anything.
- cache_01_fast_broken_fixed.c Same as cache_01_fast_broken_fixed.c, but now we do something with the result of the calculation (e.g., print the sum of the elements of y), so that the compiler does not eliminate the code. Alternatively, you can remove DCE by adding the following flags when compiling: -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse
- cache_fs_slow.c: It shows the false sharing problem
- cache_fs_fast.c: It solves the false sharing problem by padding the structure
- branch_prediction_slow.c: Fills an array with random elements between 0 and 9, and then counts the number of elements that are greater than 5. The program is slow because the branch predictor is not able to predict the outcome of the if statement.
- branch_prediction_fast.c: Same as branch_prediction_slow.c, but it sorts the array before doing the check, so that (approximatively), the first n/2 elements are smaller an the remaining n/2 larger. In this way, the branch predictor is able to predict the outcome of the if statement more effectively.


For more examples, check https://github.com/Kobzol/hardware-effects

To install perf on WSL2:
```bash
sudo apt install linux-tools-generic
```