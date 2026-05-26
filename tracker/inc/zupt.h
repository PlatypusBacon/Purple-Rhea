#ifndef PURPLE_RHEA_ZUPT_H
#define PURPLE_RHEA_ZUPT_H

#include <stdbool.h>

void zupt_init(void);

/* Feed every IMU sample (raw accel + raw gyro in body frame, m/s^2 and rad/s).
 * Updates the still-detector and, when still, tracks a low-passed gyro bias. */
void zupt_feed(float ax, float ay, float az,
	       float gx, float gy, float gz,
	       float dt);

bool zupt_is_still(void);

/* Latest gyro-bias estimate in rad/s. Subtract from raw gyro before fusion. */
void zupt_get_bias(float bias[3]);

#endif
