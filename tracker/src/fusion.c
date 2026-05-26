/* Mahony 6-axis (accel + gyro) complementary filter.
 *
 * Reference: Mahony, Hamel, Pflimlin, "Nonlinear Complementary Filters on the
 * Special Orthogonal Group" (2008). 6-axis version: only the gravity vector
 * is used to correct roll/pitch; yaw drifts open-loop and must be zeroed at
 * a known reference (see fusion_zero_yaw). */

#include "fusion.h"

#include <math.h>

#define TWO_KP (2.0f * 1.5f)   /* proportional gain — tune per IMU noise */
#define TWO_KI (2.0f * 0.005f) /* integral gain — small to reject bias drift */

static float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
static float ix, iy, iz;

static float inv_sqrt(float x)
{
	return 1.0f / sqrtf(x);
}

void fusion_init(void)
{
	q0 = 1.0f;
	q1 = q2 = q3 = 0.0f;
	ix = iy = iz = 0.0f;
}

void fusion_update(float gx, float gy, float gz,
		   float ax, float ay, float az,
		   float dt)
{
	float norm = ax * ax + ay * ay + az * az;
	if (norm > 0.0f) {
		float recip = inv_sqrt(norm);
		ax *= recip;
		ay *= recip;
		az *= recip;

		/* Estimated gravity direction from current attitude */
		float vx = 2.0f * (q1 * q3 - q0 * q2);
		float vy = 2.0f * (q0 * q1 + q2 * q3);
		float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

		/* Error is cross product between measured and estimated gravity */
		float ex = (ay * vz - az * vy);
		float ey = (az * vx - ax * vz);
		float ez = (ax * vy - ay * vx);

		if (TWO_KI > 0.0f) {
			ix += TWO_KI * ex * dt;
			iy += TWO_KI * ey * dt;
			iz += TWO_KI * ez * dt;
			gx += ix;
			gy += iy;
			gz += iz;
		}

		gx += TWO_KP * ex;
		gy += TWO_KP * ey;
		gz += TWO_KP * ez;
	}

	gx *= 0.5f * dt;
	gy *= 0.5f * dt;
	gz *= 0.5f * dt;

	float qa = q0, qb = q1, qc = q2;
	q0 += (-qb * gx - qc * gy - q3 * gz);
	q1 += ( qa * gx + qc * gz - q3 * gy);
	q2 += ( qa * gy - qb * gz + q3 * gx);
	q3 += ( qa * gz + qb * gy - qc * gx);

	float recip = inv_sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
	q0 *= recip;
	q1 *= recip;
	q2 *= recip;
	q3 *= recip;
}

void fusion_get_euler(float *roll, float *pitch, float *yaw)
{
	const float rad2deg = 57.29577951308232f;
	*roll  = atan2f(2.0f * (q0 * q1 + q2 * q3),
			1.0f - 2.0f * (q1 * q1 + q2 * q2)) * rad2deg;
	float sinp = 2.0f * (q0 * q2 - q3 * q1);
	if (sinp > 1.0f)  sinp = 1.0f;
	if (sinp < -1.0f) sinp = -1.0f;
	*pitch = asinf(sinp) * rad2deg;
	*yaw   = atan2f(2.0f * (q0 * q3 + q1 * q2),
			1.0f - 2.0f * (q2 * q2 + q3 * q3)) * rad2deg;
}

void fusion_get_quaternion(float q[4])
{
	q[0] = q0;
	q[1] = q1;
	q[2] = q2;
	q[3] = q3;
}

void fusion_zero_yaw(void)
{
	/* Rotate current attitude by -current_yaw about world Z. Compose
	 * q' = q_yaw_inv * q, where q_yaw_inv is rotation by -yaw about Z. */
	float yaw_rad = atan2f(2.0f * (q0 * q3 + q1 * q2),
			       1.0f - 2.0f * (q2 * q2 + q3 * q3));
	float half = -0.5f * yaw_rad;
	float cw = cosf(half);
	float sw = sinf(half);
	/* q_yaw_inv = (cw, 0, 0, sw) */
	float n0 = cw * q0 - sw * q3;
	float n1 = cw * q1 - sw * q2;
	float n2 = cw * q2 + sw * q1;
	float n3 = cw * q3 + sw * q0;
	q0 = n0; q1 = n1; q2 = n2; q3 = n3;
}
