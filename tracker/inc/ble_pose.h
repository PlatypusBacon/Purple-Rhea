#pragma once
#include <stdbool.h>
#include <stdint.h>

/* ble_pose.h */
void ble_pose_init(void);
void ble_pose_update(float pitch_deg, uint32_t frame_index);