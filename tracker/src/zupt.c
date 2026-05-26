/* Zero-velocity update detector + gyro-bias estimator.
 *
 * Detector: ring buffer of recent accel-norm samples; if peak-to-peak stays
 * within a tight window AND mean is near 1 g for a sustained period, we are
 * "still". While still, gyro readings are low-pass-filtered into the bias
 * estimate. */

#include "zupt.h"

#include <math.h>

#define WINDOW       32      /* ~0.31 s at 104 Hz */
#define G_MIN        9.4f
#define G_MAX        10.2f
#define P2P_THRESH   0.35f   /* m/s^2 — accel magnitude variation while still */
#define BIAS_TAU_S   0.5f    /* bias low-pass time constant */

static float ring[WINDOW];
static int ring_idx;
static int ring_filled;

static float bias_x, bias_y, bias_z;
static bool still;

void zupt_init(void)
{
	ring_idx = 0;
	ring_filled = 0;
	bias_x = bias_y = bias_z = 0.0f;
	still = false;
	for (int i = 0; i < WINDOW; i++) {
		ring[i] = 0.0f;
	}
}

void zupt_feed(float ax, float ay, float az,
	       float gx, float gy, float gz,
	       float dt)
{
	float norm = sqrtf(ax * ax + ay * ay + az * az);
	ring[ring_idx] = norm;
	ring_idx = (ring_idx + 1) % WINDOW;
	if (ring_filled < WINDOW) {
		ring_filled++;
	}

	if (ring_filled < WINDOW) {
		still = false;
		return;
	}

	float lo = ring[0], hi = ring[0], sum = 0.0f;
	for (int i = 0; i < WINDOW; i++) {
		float v = ring[i];
		if (v < lo) lo = v;
		if (v > hi) hi = v;
		sum += v;
	}
	float mean = sum / (float)WINDOW;

	still = ((hi - lo) < P2P_THRESH) && (mean > G_MIN) && (mean < G_MAX);

	if (still) {
		/* Exponential low-pass towards current gyro reading. */
		float alpha = dt / (BIAS_TAU_S + dt);
		bias_x += alpha * (gx - bias_x);
		bias_y += alpha * (gy - bias_y);
		bias_z += alpha * (gz - bias_z);
	}
}

bool zupt_is_still(void)
{
	return still;
}

void zupt_get_bias(float bias[3])
{
	bias[0] = bias_x;
	bias[1] = bias_y;
	bias[2] = bias_z;
}
