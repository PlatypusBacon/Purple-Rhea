#include "processing_thread.h"

#include "imu.h"
#include "fusion.h"
#include "zupt.h"
#include "radius.h"

#include <stdio.h>
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

	if (imu_init() != 0) {
		LOG_ERR("imu_init failed — thread exiting");
		return;
	}
	fusion_init();
	zupt_init();
	radius_init();

	const float dt = 1.0f / (float)SAMPLE_HZ;
	int settle = 0;
	bool yaw_zeroed = false;
	int emit_counter = 0;

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

			float r;
			if (radius_estimate(&r)) {
				printk("RADIUS:%.4f\n", (double)r);
			} else {
				float w, ah, omega;
				radius_status(&omega, &ah, &w);
				printk("RADIUS_DBG:omega=%.3f a_horiz=%.3f weight=%.4f\n",
				       (double)omega, (double)ah, (double)w);
			}
		}
	}
}
