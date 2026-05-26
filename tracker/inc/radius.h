#ifndef PURPLE_RHEA_RADIUS_H
#define PURPLE_RHEA_RADIUS_H

#include <stdbool.h>

void radius_init(void);

/* Feed one IMU tick after gyro-bias correction.
 *
 *   q          body-to-world quaternion (w, x, y, z) from fusion
 *   accel_body raw accel in m/s^2, body frame
 *   gyro_body  bias-corrected gyro in rad/s, body frame
 *   dt         tick period in seconds
 *
 * Internally: rotate accel and gyro into world frame, subtract gravity from
 * world-frame accel, accumulate (omega^2, |a_horiz|) for weighted least
 * squares. Samples are gated on |omega| > omega_min and steady rotation.
 */
void radius_feed(const float q[4],
		 const float accel_body[3],
		 const float gyro_body[3],
		 float dt);

/* Latest estimate. Returns true and writes r in metres if enough samples
 * have accumulated; false if not yet converged. */
bool radius_estimate(float *r_out);

/* Diagnostic snapshot — last accepted (omega_z, a_horiz) and accumulated
 * weight. Useful for debugging why the estimator isn't converging. */
void radius_status(float *omega_z, float *a_horiz, float *weight);

#endif
