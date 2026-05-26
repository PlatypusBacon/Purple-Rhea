#ifndef PURPLE_RHEA_FUSION_H
#define PURPLE_RHEA_FUSION_H

void fusion_init(void);

/* gyro in rad/s, accel in any consistent unit (only direction matters). */
void fusion_update(float gx, float gy, float gz,
		   float ax, float ay, float az,
		   float dt);

/* Output Euler angles in degrees (roll, pitch, yaw, ZYX convention). */
void fusion_get_euler(float *roll, float *pitch, float *yaw);

/* Output unit quaternion (w, x, y, z) representing body-to-world rotation. */
void fusion_get_quaternion(float q[4]);

/* Force yaw to zero at the current orientation (rotates the reference frame
 * about gravity). Use at scan start with the arm at the 0° position. */
void fusion_zero_yaw(void);

#endif
