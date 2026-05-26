/* Online weighted least-squares estimator for the orbit radius of a rigid
 * arm spinning about a vertical axis.
 *
 * Physics: a point at radius r from the rotation axis spinning at angular
 * velocity ω experiences centripetal acceleration of magnitude ω²·r in the
 * horizontal plane, pointed at the axis. Given the fusion's body-to-world
 * quaternion we can rotate the body-frame accel into the world frame,
 * subtract gravity, and use the magnitude of the horizontal residual.
 *
 * Likewise, the gyro vector rotated into the world frame gives the angular
 * velocity about world-Z directly — independent of how the IMU is mounted.
 *
 * Weighted LSQ slope through origin: y_i = a_horiz_i, x_i = ω_z_world².
 *   r̂ = Σ(x_i · y_i) / Σ(x_i²)
 * Exponentially-weighted accumulators give a slowly-adapting estimate. */

#include "radius.h"

#include <math.h>

#define GRAVITY        9.80665f
#define OMEGA_MIN      0.15f     /* rad/s — ~8.6°/s, comfortable hand-spin */
#define DOMEGA_MAX     15.0f     /* rad/s² — reject only large tangential spikes */
#define TAU_S          5.0f      /* averaging time constant */
#define MIN_WEIGHT_SUM 0.05f     /* Σω⁴ threshold to publish an estimate */

static float s_xy;             /* Σ ω² · a_horiz */
static float s_xx;             /* Σ ω⁴ */
static float prev_omega_z;
static bool  prev_omega_valid;
static float last_omega_z;     /* most recent ω (any value, ungated) */
static float last_a_horiz;

void radius_init(void)
{
	s_xy = 0.0f;
	s_xx = 0.0f;
	prev_omega_z = 0.0f;
	prev_omega_valid = false;
	last_omega_z = 0.0f;
	last_a_horiz = 0.0f;
}

/* Rotate a body-frame vector v into the world frame using unit quaternion q
 * (w, x, y, z). Uses v' = v + 2·w·(q_vec × v) + 2·(q_vec × (q_vec × v)). */
static void rotate_body_to_world(const float q[4], const float v[3], float out[3])
{
	float w = q[0], x = q[1], y = q[2], z = q[3];

	float tx = 2.0f * (y * v[2] - z * v[1]);
	float ty = 2.0f * (z * v[0] - x * v[2]);
	float tz = 2.0f * (x * v[1] - y * v[0]);

	out[0] = v[0] + w * tx + (y * tz - z * ty);
	out[1] = v[1] + w * ty + (z * tx - x * tz);
	out[2] = v[2] + w * tz + (x * ty - y * tx);
}

void radius_feed(const float q[4],
		 const float accel_body[3],
		 const float gyro_body[3],
		 float dt)
{
	float a_world[3];
	float g_world[3];
	rotate_body_to_world(q, accel_body, a_world);
	rotate_body_to_world(q, gyro_body, g_world);

	/* Subtract gravity in world frame. */
	a_world[2] -= GRAVITY;

	float omega_z = g_world[2];
	float a_horiz = sqrtf(a_world[0] * a_world[0] + a_world[1] * a_world[1]);

	last_omega_z = omega_z;
	last_a_horiz = a_horiz;

	/* Gate: need sustained, steady spin. */
	if (fabsf(omega_z) < OMEGA_MIN) {
		prev_omega_valid = false;
		return;
	}
	if (prev_omega_valid) {
		float domega = (omega_z - prev_omega_z) / dt;
		if (fabsf(domega) > DOMEGA_MAX) {
			prev_omega_z = omega_z;
			return;
		}
	}
	prev_omega_z = omega_z;
	prev_omega_valid = true;

	/* Exponential weighting: each new sample contributes (1 - α). */
	float alpha = expf(-dt / TAU_S);
	float x = omega_z * omega_z;

	s_xy = alpha * s_xy + (1.0f - alpha) * (x * a_horiz);
	s_xx = alpha * s_xx + (1.0f - alpha) * (x * x);
}

bool radius_estimate(float *r_out)
{
	if (s_xx < MIN_WEIGHT_SUM) {
		return false;
	}
	float r = s_xy / s_xx;
	if (!isfinite(r) || r < 0.0f) {
		return false;
	}
	*r_out = r;
	return true;
}

void radius_status(float *omega_z, float *a_horiz, float *weight)
{
	if (omega_z) *omega_z = last_omega_z;
	if (a_horiz) *a_horiz = last_a_horiz;
	if (weight)  *weight  = s_xx;
}
