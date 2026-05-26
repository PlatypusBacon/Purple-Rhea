#pragma once
#include <stdbool.h>
#include <stdint.h>

void ble_pose_init(void);

/* Call every sample tick to keep cached pose fresh */
void ble_pose_update(float roll, float pitch, float yaw,
                     float radius, bool radius_valid, uint32_t frame_index);