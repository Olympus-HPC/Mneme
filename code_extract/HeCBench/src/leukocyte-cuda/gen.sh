rm -rf dpct_output
dpct detect_main.c track_ellipse_gpu.c --extra-arg="-I./meschach_lib"  --usm-level=none
cp avilib.h avilib.c track_ellipse.c find_ellipse.c helper.c misc_math.c dpct_output
cd dpct_output
rm -f *.yaml
rename 's/\.c.dp.cpp/\.c/' *.dp.cpp 
rm -rf meschach_lib
cp -r ../../leukocyte-sycl/lmeschach_lib .
