cmake_minimum_required(VERSION 3.16)
project(rk_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 构建产物直接采用 INSTALL_RPATH，不让 CMake 另拼一份构建树 RPATH。
# 默认（OFF）时 CMake 会把 target_link_directories 里的**绝对路径**写进
# RUNPATH —— 实测产物里残留过 /root/PyVideoProc_RK/kkk/cpp/third_party/mpp/lib
# 这种条目，目录一改名或一拷到新板子就是死路径。
# 这里没有 install() 规则：build/ 就是交付位置，产物就地 import。
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# pybind11 走 .venv（pyproject.toml 的依赖，uv sync 装好），不用 apt 的 pybind11-dev：
# 新板子上少装一个 apt 包，而且 pip 的 pybind11 比 2.9 新，没有 2.9 那批
# array_t 花括号重载推导不出来的坑。用 uv 构建时 pybind11 的 config 会通过
# CMAKE_PREFIX_PATH（.venv/lib/python3.10/site-packages）被找到。
find_package(pybind11 REQUIRED)

# numpy 头文件位置：问本 venv 的 python 最稳（pybind11 的 config 会设好
# Python_EXECUTABLE；没设就退回 PATH 里的 python3）
if(NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE python3)
endif()
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_RET)
if(NOT NUMPY_RET EQUAL 0)
    message(FATAL_ERROR "${Python_EXECUTABLE} -c 'import numpy' 失败——先 uv sync")
endif()

# 复用 cpp/ 里阶段 3 实测过的积木：dma_buf 池 / 数据源 / MPP 解码 / RGA
set(CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(TP  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# 自带 librga / librockchip_mpp（third_party/ 里，与板上 BSP 同版本）还是用
# 系统 apt 装的。默认自带 → 新板子不用装 librga-dev / librockchip-mpp-dev。
# 注意：这两个是用户态库，必须与目标板内核里的 rga/mpp 驱动匹配；如果目标板
# 的 BSP 与这份自带库不同源，用 -DUSE_BUNDLED_LIBS=OFF 改链系统的。
option(USE_BUNDLED_LIBS "链接 third_party/ 自带的 librga + librockchip_mpp" ON)

pybind11_add_module(rk_decoder
    ${CPP_DIR}/rk_decoder.cpp
    ${CPP_DIR}/dmabuf.cpp
    ${CPP_DIR}/dmabuf_pool.cpp
    ${CPP_DIR}/h264_source.cpp
    ${CPP_DIR}/mpp_decoder.cpp
    ${CPP_DIR}/rga_convert.cpp)

target_include_directories(rk_decoder PRIVATE ${CPP_DIR} ${NUMPY_INCLUDE})

if(USE_BUNDLED_LIBS)
    target_include_directories(rk_decoder PRIVATE
        ${TP}/mpp/include ${TP}/rga/include)
    target_link_directories(rk_decoder PRIVATE
        ${TP}/mpp/lib ${TP}/rga/lib)
    # 自带库用 $ORIGIN 相对路径找：产物在 build/，库在 ../third_party/*/lib，
    # 这样整个目录树拷到哪都能 import，不用设 LD_LIBRARY_PATH。
    # $ORIGIN 就是 .so 自己所在的 build/，所以路径里带 ".."。
    set_target_properties(rk_decoder PROPERTIES
        INSTALL_RPATH "$ORIGIN/../third_party/mpp/lib;$ORIGIN/../third_party/rga/lib")
endif()

target_link_libraries(rk_decoder PRIVATE rga rockchip_mpp)

# 产物: core/build/rk_decoder.cpython-310-aarch64-linux-gnu.so
#       外层的 main.py / rtsp_yolo.py 会把它所在的 build/ 加进 sys.path；
#       core/ 里自带脚本（infer.py / demo.py）按 __file__ 定位。
