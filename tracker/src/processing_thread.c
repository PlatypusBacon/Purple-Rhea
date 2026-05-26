#include "processing_thread.h"
#include "imu.h"
#include "ble_pose.h"
#include <math.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

LOG_MODULE_REGISTER(processing, LOG_LEVEL_INF);

void processing_thread_entry(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);
    k_msleep(3000);

    if (imu_init() != 0) {
        LOG_ERR("imu_init failed — thread exiting");
        return;
    }

    ble_pose_init();
    printk("STATUS:ready\n");

    uint32_t frame_index = 0;

    while (true) {
        k_msleep(50);  /* 20 Hz */

        float a[3], g[3];
        if (imu_read(a, g) != 0) {
            LOG_WRN("imu_read failed");
            continue;
        }

        float ax = a[0], ay = a[1], az = a[2];
        float horiz = sqrtf(ax * ax + ay * ay);
        float pitch_deg = atan2f(-az, horiz) * (180.0f / (float)M_PI);

        ble_pose_update(pitch_deg, frame_index++);
    }
}