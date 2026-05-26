#include "processing_thread.h"

#include "imu.h"
#include "fusion.h"
#include "zupt.h"
#include "radius.h"
#include "tracker_pose.pb.h"
#include <pb_encode.h>
#include <zephyr/drivers/uart.h>
#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(processing, LOG_LEVEL_INF);

/* 104 Hz IMU sample → fusion + ZUPT, 20 Hz LOCATION emission on CDC console.
 * Yaw auto-zeroes the first time the rig is held still after a short settle. */

#define SAMPLE_HZ        104
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_HZ)
#define EMIT_EVERY_N     5   /* 104/5 ≈ 20 Hz LOCATION emission */
#define SETTLE_TICKS     (SAMPLE_HZ * 2)   /* wait 2 s before allowing auto-zero */



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

        /* a[] is [x, y, z] in sensor frame.
         * Define pitch_deg as the angle of the USB axis from horizontal.
         * 0°  = USB pointing horizontally away from centre
         * 90° = USB pointing straight down at ground
         * Adjust axis sign/index to match your board orientation. */
        float ax = a[0], ay = a[1], az = a[2];
        float horiz = sqrtf(ax * ax + ay * ay);
        float pitch_deg = atan2f(-az, horiz) * (180.0f / (float)M_PI);

        ble_pose_update(pitch_deg, frame_index++);
    }
}
