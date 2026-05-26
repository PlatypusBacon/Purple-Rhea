#include "imu.h"

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(imu, LOG_LEVEL_INF);

/* LSM6DS3TR-C on XIAO BLE Sense is declared with compatible "st,lsm6dsl" in
 * the board DTS, so DEVICE_DT_GET_ONE(st_lsm6dsl) resolves to it. */
static const struct device *imu_dev;

int imu_init(void)
{
	imu_dev = DEVICE_DT_GET_ONE(st_lsm6dsl);
	if (!device_is_ready(imu_dev)) {
		LOG_ERR("lsm6dsl not ready");
		return -ENODEV;
	}

	struct sensor_value odr = { .val1 = 104, .val2 = 0 };

	if (sensor_attr_set(imu_dev, SENSOR_CHAN_ACCEL_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr) < 0) {
		LOG_ERR("set accel ODR failed");
		return -EIO;
	}
	if (sensor_attr_set(imu_dev, SENSOR_CHAN_GYRO_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr) < 0) {
		LOG_ERR("set gyro ODR failed");
		return -EIO;
	}

	LOG_INF("IMU ready @ 104 Hz");
	return 0;
}

int imu_read(float accel[3], float gyro[3])
{
	struct sensor_value v[3];

	if (sensor_sample_fetch(imu_dev) < 0) {
		return -EIO;
	}

	if (sensor_channel_get(imu_dev, SENSOR_CHAN_ACCEL_XYZ, v) < 0) {
		return -EIO;
	}
	accel[0] = (float)sensor_value_to_double(&v[0]);
	accel[1] = (float)sensor_value_to_double(&v[1]);
	accel[2] = (float)sensor_value_to_double(&v[2]);

	if (sensor_channel_get(imu_dev, SENSOR_CHAN_GYRO_XYZ, v) < 0) {
		return -EIO;
	}
	gyro[0] = (float)sensor_value_to_double(&v[0]);
	gyro[1] = (float)sensor_value_to_double(&v[1]);
	gyro[2] = (float)sensor_value_to_double(&v[2]);

	return 0;
}
