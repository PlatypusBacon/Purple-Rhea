#ifndef PURPLE_RHEA_IMU_H
#define PURPLE_RHEA_IMU_H

#include <stdbool.h>

int imu_init(void);

/* Returns 0 on success. accel in m/s^2, gyro in rad/s. */
int imu_read(float accel[3], float gyro[3]);

#endif
