cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(base_node)

target_sources(app PRIVATE
    src/main.c
    src/processing_thread.c
    src/imu.c
    src/fusion.c
    src/zupt.c
    src/radius.c
    src/tracker_pose.pb.c
    src/ble_pose.c
)
zephyr_include_directories(
    inc
    ../inc
)

target_link_libraries(app PRIVATE m)