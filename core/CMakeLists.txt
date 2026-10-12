cmake_minimum_required(VERSION 3.16)
project(rk_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)  # 构建产物直接采用 INSTALL_RPATH

# ---------------------------------------------------------------- Python 侧
# 解释器走 .venv
set(VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
if(NOT EXISTS "${VENV_PYTHON}")
    message(WARNING "没找到 ${VENV_PYTHON}，退回系统 python3 —— 先 uv sync")
    set(VENV_PYTHON python3)
endif()
set(PYTHON_EXECUTABLE "${VENV_PYTHON}" CACHE FILEPATH "Python 解释器（项目 .venv）" FORCE)

# pybind11 走 .venv 不用 apt 的 pybind11-dev
file(GLOB _venv_pybind11 "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.*/site-packages/pybind11")
if(_venv_pybind11)
    list(GET _venv_pybind11 0 _venv_pybind11)
    list(PREPEND CMAKE_PREFIX_PATH "${_venv_pybind11}")
endif()

set(PYBIND11_FINDPYTHON ON)  # 走新的 FindPython，别用被 CMP0148 废弃的老 FindPythonLibs
find_package(pybind11 REQUIRED)  # 在 .venv 里寻找

# numpy 头文件位置：问同一个解释器要，保证编译期头文件与运行期 numpy 同一份
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_RET)
if(NOT NUMPY_RET EQUAL 0)
    message(FATAL_ERROR "${PYTHON_EXECUTABLE} -c 'import numpy' error, please uv sync")
endif()

# ---------------------------------------------------------------- MPP / RGA
#
# 用目标板自带的，不自带副本。
#
# 这两个是 userspace 库，必须和板子内核里的 mpp/rga 驱动配对；厂商是把
# userspace 包和内核驱动一起发布的 —— 用板子自带的天然配对。自带一份反而
# 会引入"userspace 与内核驱动不同源"的错配，症状是花屏或段错误，极难查。
# pkg-config 让换板子时自动适配，不用改路径。
#
# 板子上没装的话这里就失败，报错比链接器甩一堆 "cannot find -lrockchip_mpp" 清楚：
#   sudo apt install librockchip-mpp-dev librga-dev
find_package(PkgConfig REQUIRED)
pkg_check_modules(MPP REQUIRED rockchip_mpp)
pkg_check_modules(RGA REQUIRED librga)

# --------------------------------------------------- 能力检测（不是版本号比较）
#
# 为什么不比版本号：这个平台的版本号体系是乱的 —— 同一个 MPP，pkg-config
# 报 1.3.8 而 dpkg 报 1.5.0-2；同一个 RGA 有三个号：包 2.2.0-1 / 文件名
# 2.1.0 / 运行时 API 1.9.3_[2]。拿版本号比大小根本对不上。
#
# 直接编译一段"用到目标 API"的代码，能不能过就是准的 —— 这才是真正要问的问题：
# "这块板子支不支持我代码里用到的东西"。
#
# 注意方向：Rockchip 的向后兼容是"新库兼容老代码"（头文件里同时留着新老两套
# API），**不保证老库支持新代码**。而本代码用的恰恰是新 API（MppDecCfg 是
# 2020 年才有的、improcess 10 参重载也是新的），所以老板子上必然失败 ——
# 失败要趁早，而且要报人话。
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${MPP_INCLUDE_DIRS} ${RGA_INCLUDE_DIRS})

set(CMAKE_REQUIRED_LIBRARIES ${MPP_LIBRARIES})
check_cxx_source_compiles([[
#include <rockchip/rk_mpi.h>
#include <rockchip/rk_vdec_cfg.h>
int main() {
    MppCtx ctx = NULL; MppApi *mpi = NULL;
    if (mpp_create(&ctx, &mpi)) return 1;
    mpi->control(ctx, MPP_SET_DISABLE_THREAD, NULL);   /* 关内部线程，2020 年后才有 */
    MppDecCfg cfg = NULL;
    mpp_dec_cfg_init(&cfg);                            /* MppDecCfg API，2020 年后才有 */
    mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);   /* 任意块大小直接喂，靠它 */
    mpp_dec_cfg_deinit(cfg);
    mpp_destroy(ctx);
    return 0;
}
]] HAVE_MPP_REQUIRED_API)
unset(CMAKE_REQUIRED_LIBRARIES)

if(NOT HAVE_MPP_REQUIRED_API)
    message(FATAL_ERROR
        "\n这块板子的 MPP 太老，缺本代码需要的 API。\n"
        "  需要: MppDecCfg (mpp_dec_cfg_init/set_u32/deinit) + MPP_SET_DISABLE_THREAD\n"
        "  也就是 MPP >= 1.0.6 (2021)。\n"
        "  pkg-config 报的版本: ${MPP_VERSION}（这个号不可靠，以编译结果为准）\n"
        "  → 升级板子的 librockchip-mpp，或换一块 BSP 更新的板子。\n")
endif()

set(CMAKE_REQUIRED_LIBRARIES ${RGA_LIBRARIES})
check_cxx_source_compiles([[
#include <cstddef>          /* im2d.h 里用了 NULL 却没包 cstddef，得先垫上 */
#include <rga/im2d.hpp>     /* 10 参 C++ 重载声明在 .hpp 里，.h 只是 C 版 7 参，且不会自动 include .hpp */
#include <rga/rga.h>        /* RK_FORMAT_* 枚举定义在这两个都不会自动带入 */
int main() {
    /* 本代码依赖的两个新 API，各自测一遍（都实测过、不是照文档猜的）：
     *   1. wrapbuffer_fd_t 带 w/h/stride 的 6 参版本（老版本只有 2 参）
     *   2. improcess 的 10 参 C++ 重载 —— im2d_single.h 里同名两个：
     *        442 行 IM_API   10 参，第 7 个是 acquire_fence_fd  ← 我们要的
     *        479 行 IM_C_API  7 参，第 7 个是 usage
     *      只测名字测不出来，必须按 10 参调用才能逼编译器选中那个重载。
     * 不测 RGA_CURRENT_API_HEADER_VERSION：那是头文件宏，只反映头文件、
     * 反映不了 .so；且万一新版本把宏删了，预处理按 0 算 → 好板子误报。
     * 运行期只打个版本日志（见 rga_convert.cpp 的 rga_check_api）。 */
    rga_buffer_t src = wrapbuffer_fd_t(0, 16, 16, 16, 16, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_fd_t(0, 16, 16, 16, 16, RK_FORMAT_RGB_888);
    rga_buffer_t pat = {};
    im_rect sr = { 0, 0, 16, 16 }, dr = { 0, 0, 16, 16 }, pr = { 0, 0, 0, 0 };
    IM_STATUS st = improcess(src, dst, pat, sr, dr, pr,
                             0, nullptr, nullptr, IM_SYNC);
    (void)st;
    return 0;
}
]] HAVE_RGA_REQUIRED_API)
unset(CMAKE_REQUIRED_LIBRARIES)

if(NOT HAVE_RGA_REQUIRED_API)
    message(FATAL_ERROR
        "\n这块板子的 RGA im2d API 太老，缺本代码需要的 API。\n"
        "  需要: wrapbuffer_fd_t(带 w/h/stride 的 6 参版本)\n"
        "      + improcess 的 10 参 C++ 重载（第 7 个参数是 acquire_fence_fd）\n"
        "  大致对应 im2d API >= 1.9.3。\n"
        "  pkg-config 报的版本: ${RGA_VERSION}（这个号不可靠，以编译结果为准）\n"
        "  → 升级板子的 librga，或换一块 BSP 更新的板子。\n")
endif()

# ---------------------------------------------------------------- 目标
set(CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

pybind11_add_module(rk_decoder
    ${CPP_DIR}/rk_decoder.cpp
    ${CPP_DIR}/dmabuf.cpp
    ${CPP_DIR}/dmabuf_pool.cpp
    ${CPP_DIR}/h264_source.cpp
    ${CPP_DIR}/mpp_decoder.cpp
    ${CPP_DIR}/rga_convert.cpp)

target_include_directories(rk_decoder PRIVATE
    ${CPP_DIR} ${NUMPY_INCLUDE} ${MPP_INCLUDE_DIRS} ${RGA_INCLUDE_DIRS})

target_link_directories(rk_decoder PRIVATE ${MPP_LIBRARY_DIRS} ${RGA_LIBRARY_DIRS})

# 不再设 INSTALL_RPATH：mpp/rga 走系统路径（ldconfig），不用 $ORIGIN 找自带副本。
# 想临时改用别的库，LD_LIBRARY_PATH 优先于 RUNPATH，不用重编。
target_link_libraries(rk_decoder PRIVATE ${MPP_LIBRARIES} ${RGA_LIBRARIES})
