Lab 10 - CUDA

Student ID - IT24610798

GPU - Tesla T4





Exercise 01 - Timing a GPU program



Experiment: removing cudaDeviceSynchronize()

&#x09;- With cudaDeviceSynchronize(): Time taken = 0.0008650 s

&#x09;- Without it: Time taken = 0.0001011 s

&#x09;- Verification still reported 0 mismatches in both cases.



Explanation: A kernel launch is asynchronous. The CPU starts the GPU and carries on immediately. Without cudaDeviceSynchronize(), the clock stopped straight after the launch, so it only measured the launch and not the kernel running. The answer still verified correctly because the later cudaMemcpy (device to host) waits for the GPU to finish before copying.





Part (b) – timing table



|Lab 9 program|Launch configuration|Run 1 (s)|Run 2 (s)|Run 3 (s)|
|-|-|-|-|-|
|Exercise 3 |512 blocks x 1 thread|0.0000989|0.0001159|0.0001159|
|Exercise 4 |1 block x 512 threads|0.0000961|0.0001481|0.0000930|
|Exercise 5|10,000,000 elements,|0.0006211|0.0006030|0.0006051|
|Exercise 6|100,000,000 elements|0.0060148|0.0050318|0.0050230|



Prediction 1 (Ex 3 vs Ex 4):The two are about the same (around 0.1 ms). With only 512 elements the kernel does almost no work, so launch and synchronise overhead dominate, and the run-to-run noise is bigger than any difference between the two layouts.



Prediction 2 (Ex 5 vs Ex 6):Ex 6 has 10 times as many elements, but its average time (about 0.0054 s) is only about 8.8 times Ex 5 (about 0.0006 s). The kernel is limited by memory speed, so time grows roughly in line with data size, and fixed launch overhead is shared, so it is slightly less than 10 times.





Part (c) – kernel time vs copy time (100 million elements)



&#x09;- Kernel only: 0.0047431 s

&#x09;- Whole sequence (copy in + kernel + copy out): 0.4376781 s





The whole sequence is about 92 times larger than the kernel alone. The arrays are about 400 MB each and must travel over the PCIe bus in both directions, and the multiply is only one operation per element. So copying dominates. A GPU is worth using when there is a lot of computation per byte of data moved.





Exercise 2 – Catching errors with the CHECK() macro



All five TODOs were wrapped in CHECK(). The output matched Exercise 1 (0 mismatches out of 512).



Experiment A – too many threads (launch changed to <<<N, 2048>>>):

Output: `Error: ex2\_check.cu:<line>, code:9, reason: invalid configuration argument`

This tested the kernel launch row (caught by CHECK(cudaGetLastError())). A block can have at most 1024 threads.



Experiment B – impossible allocation ((size\_t)1 << 50 bytes):

Output: `Error: ex2\_check.cu:39, code:2, reason: out of memory`

This tested the "CUDA call that returns a value" row (cudaMalloc wrapped in CHECK).





Experiment C – no CHECK (Experiment A on the plain Exercise 1 program):

No error message was printed. The launch failed silently and only the verification loop showed mismatches. Without the verification loop, we would not know anything had gone wrong. This is why CHECK() is useful.



The third row (kernel fails while running, e.g. illegal memory access) is caught by CHECK(cudaDeviceSynchronize()).





Exercise 3 – Scaling a picture with a 2D grid



&#x09;- The picture is a one-dimensional array of 800 x 600 = 480,000 unsigned chars, 	each between 0 and 255.

&#x09;- Launch: 16 x 16 threads per block, grid of 50 x 38 blocks (dim3 variables).

&#x09;- 800 / 16 = 50 blocks in x (exact fit). 600 / 16 = 37.5, rounded up to 38 blocks 	in y. The last block row has 8 rows inside the picture and 8 outside, so the guard 	(Row < height \&\& Col < width) stops those threads.



Prediction: about half of the pixels would be capped at 255, because any original value from 128 to 255 becomes 255 when doubled (128 of 256 possible values).



Result:

&#x09;- Verification: 0 mismatches out of 480000

&#x09;- Pixels capped at 255: 240121

&#x09;- Time taken: 0.0001609 s





Exercise 4 – Multiplying two 3D matrices element by element



\- Volume size: 500 x 500 x 500 = 125,000,000 elements (500 MB per matrix).

\- Launch: 8 x 8 x 8 = 512 threads per block, grid of 63 x 63 x 63 blocks (500 / 8 = 62.5, rounded up).

\- Guard: x < nx \&\& y < ny \&\& z < nz. Flat index: (z \* ny + y) \* nx + x.

\- Last result: C\[499]\[499]\[499] = 47 x 68 = 3196

\- Verification: 0 mismatches out of 125000000

\- Time taken: \[your Exercise 4 time] s



THINK – 3D vs 2D:

The 3D kernel takes about the same time as the 2D kernel in Exercise 6 (\[your Ex 6 average, about 0.0054 s]), a little more because it has 25% more elements. The work per element is the same (two reads, one multiply, one write), so the kernel is memory-bound, not compute-bound. The thread arrangement matters little because neighbouring threads (x changes fastest) read neighbouring memory addresses, so memory access stays efficient in 1D, 2D or 3D. It would matter a lot if neighbouring threads accessed far-apart addresses, because memory access would then be slow.

