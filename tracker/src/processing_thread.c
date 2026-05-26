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


#define UART_DEVICE_NODE DT_NODELABEL(uart1)
static const struct device *uart_dev = DEVICE_DT_GET(UART_DEVICE_NODE);

/* Frame format: [0xAA][len_hi][len_lo][proto payload][0x55] */
static void send_pose_uart(float roll, float pitch, float yaw,
                           float radius, bool radius_valid,
                           uint32_t frame_idx)
{
    if (!device_is_ready(uart_dev)) {
        LOG_ERR("UART device not ready");
        return;
    }

    TrackerPose msg = TrackerPose_init_zero;
    msg.yaw          = yaw;
    msg.pitch        = pitch;
    msg.roll         = roll;
    msg.frame_index  = frame_idx;
    msg.radius       = radius;
    msg.radius_valid = radius_valid;

    /* Encode into a stack buffer */
    uint8_t proto_buf[TrackerPose_size];
    pb_ostream_t stream = pb_ostream_from_buffer(proto_buf, sizeof(proto_buf));
    if (!pb_encode(&stream, TrackerPose_fields, &msg)) {
        LOG_ERR("nanopb encode failed: %s", PB_GET_ERROR(&stream));
        return;
    }

    uint16_t proto_len = (uint16_t)stream.bytes_written;

    /* Build framed packet: header + payload + sentinel */
    uint8_t frame[3 + TrackerPose_size + 1];
    frame[0] = 0xAA;
    frame[1] = (proto_len >> 8) & 0xFF;   /* len high byte */
    frame[2] =  proto_len       & 0xFF;   /* len low byte  */
    memcpy(&frame[3], proto_buf, proto_len);
    frame[3 + proto_len] = 0x55;

    uint16_t total = 4 + proto_len;
    for (uint16_t b = 0; b < total; b++) {
        uart_poll_out(uart_dev, frame[b]);
    }
}

void processing_thread_entry(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);
	k_msleep(3000);
	if (imu_init() != 0) {
		LOG_ERR("imu_init failed — thread exiting");
		return;
	}
    LOG_INF("uart1 ready");
	fusion_init();
	zupt_init();
	ble_pose_init();
	radius_init();

	const float dt = 1.0f / (float)SAMPLE_HZ;
	int settle = 0;
	bool yaw_zeroed = false;
	int emit_counter = 0;
	uint32_t frame_index = 0;

	printk("STATUS:ready\n");

	while (true) {
		k_msleep(SAMPLE_PERIOD_MS);

		float a[3], g[3];
		if (imu_read(a, g) != 0) {
			LOG_WRN("imu_read failed");
			continue;
		}

		zupt_feed(a[0], a[1], a[2], g[0], g[1], g[2], dt);

		float bias[3];
		zupt_get_bias(bias);

		float g_corr[3] = { g[0] - bias[0], g[1] - bias[1], g[2] - bias[2] };

		fusion_update(g_corr[0], g_corr[1], g_corr[2],
			      a[0], a[1], a[2], dt);

		if (settle < SETTLE_TICKS) {
			settle++;
		} else if (!yaw_zeroed && zupt_is_still()) {
			fusion_zero_yaw();
			yaw_zeroed = true;
			LOG_INF("yaw auto-zeroed");
		}

		/* Feed radius estimator after settle. Gravity subtraction depends
		 * on roll/pitch tracking gravity (not yaw), so we don't need
		 * yaw_zeroed here. ZUPT-still windows have ω≈0 which the
		 * estimator naturally rejects via its own gate. */
		if (settle >= SETTLE_TICKS) {
			float q[4];
			fusion_get_quaternion(q);
			radius_feed(q, a, g_corr, dt);
		}

		if (++emit_counter >= EMIT_EVERY_N) {
            emit_counter = 0;

            float roll, pitch, yaw;
            fusion_get_euler(&roll, &pitch, &yaw);
            printk("LOCATION:%.2f,%.2f,%.2f\n",
                (double)roll, (double)pitch, (double)yaw);

            float r = 0.0f;
            bool  r_valid = radius_estimate(&r);
            if (!r_valid) { r = 0.0f; }

            /* Update cached BLE pose — only sent when Pi requests it */
            ble_pose_update(roll, pitch, yaw, r, r_valid, frame_index++);
        }
	}
}
