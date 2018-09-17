cmake_minimum_required(VERSION 2.8)
project(fast_pcst)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython2.7 -pthread")
set(ROOT /network/rit/lab/ceashpc/bz383376/opt/)
set(NUMPY env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/)
set(OPENBLAS_INCLUDE_DIR "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_INCLUDE "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_LIB "${ROOT}openblas-0.3.1/lib/")
set(NUMPY_INCLUDE "${ROOT}${NUMPY}")
set(PYTHON_LIBRARY "${ROOT}python-2.7.14/lib/libpython2.7.so")
set(PYTHON_INCLUDE "${ROOT}python-2.7.14/include/python2.7")
include_directories(${OPENBLAS_INCLUDE_DIR})
find_package(PythonLibs 2.7 REQUIRED)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

set(SRC_PCST fast_pcst.c fast_pcst.h)
add_executable(test_fast_pcst fast_pcst_test.c ${SRC_PCST})
target_link_libraries(test_fast_pcst -std=c11 -Wall -Wextra -O3 -lm)

set(SRC_Head head_tail_proj.c head_tail_proj.h)
add_executable(test_head_tail head_tail_proj_test.c ${SRC_PCST} ${SRC_Head})
target_link_libraries(test_head_tail -std=c11 -Wall -Wextra -O3 -lm)

set(SRC_Sort sort.c sort.h)
add_executable(test_sort ${SRC_Sort} sort_test.c)
target_link_libraries(test_sort -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_metrics metrics_test.c metrics.c metrics.h ${SRC_Sort})
target_link_libraries(test_sort -std=c11 -Wall -Wextra -O3 -lm)

add_executable(test_loss loss.c loss.h loss_test.c)
target_link_libraries(test_loss -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lopenblas -lpthread)


add_executable(test_cpu_info get_cpu.c get_cpu.h)
target_link_libraries(test_cpu_info -std=c11 -Wall -Wextra -O3)

set(SRC_1 fast_pcst.c fast_pcst.h head_tail_proj.c head_tail_proj.h)
set(SRC_2 sparse_algorithms.c sparse_algorithms.h sort.c sort.h loss.c loss.h)

add_executable(test_sparse_algorithm sparse_algorithms_test.c ${SRC_1} ${SRC_2} math_utils.c math_utils.h)
target_link_libraries(test_sparse_algorithm python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

add_executable(main_wrapper main_wrapper.c ${SRC_1} ${SRC_2})
target_link_libraries(main_wrapper python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

add_library(sparse_module SHARED main_wrapper.c ${SRC_1} ${SRC_2})
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE}
        -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

