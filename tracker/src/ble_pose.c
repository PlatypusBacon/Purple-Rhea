#include "ble_pose.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/logging/log.h>

#include "tracker_pose.pb.h"
#include <pb_encode.h>

LOG_MODULE_REGISTER(ble_pose, LOG_LEVEL_INF);

#define BT_UUID_POSE_SVC_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef0)
#define BT_UUID_POSE_CHR_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef1)

static struct bt_uuid_128 pose_svc_uuid = BT_UUID_INIT_128(BT_UUID_POSE_SVC_VAL);
static struct bt_uuid_128 pose_chr_uuid = BT_UUID_INIT_128(BT_UUID_POSE_CHR_VAL);

extern const struct bt_gatt_service_static pose_svc;
static bool notify_enabled = false;

static void ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    LOG_INF("pose notify %s", notify_enabled ? "enabled" : "disabled");
}

BT_GATT_SERVICE_DEFINE(pose_svc,
    BT_GATT_PRIMARY_SERVICE(&pose_svc_uuid),
    BT_GATT_CHARACTERISTIC(&pose_chr_uuid.uuid,
                           BT_GATT_CHRC_NOTIFY,
                           BT_GATT_PERM_NONE,
                           NULL, NULL, NULL),
    BT_GATT_CCC(ccc_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR),
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_POSE_SVC_VAL),
};

static const struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, "TrackerPose", sizeof("TrackerPose") - 1),
};

void ble_pose_init(void)
{
    int err = bt_enable(NULL);
    if (err) {
        LOG_ERR("bt_enable failed: %d", err);
        return;
    }
    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd, ARRAY_SIZE(sd));
    if (err) {
        LOG_ERR("adv_start failed: %d", err);
        return;
    }
    LOG_INF("BLE advertising as 'TrackerPose'");
}

/* Called from processing_thread at 20 Hz — encodes and notifies immediately */
void ble_pose_update(float pitch_deg, uint32_t frame_index)
{
    if (!notify_enabled) return;

    TrackerPose msg = TrackerPose_init_zero;
    msg.pitch       = pitch_deg;
    msg.frame_index = frame_index;

    uint8_t buf[TrackerPose_size];
    pb_ostream_t stream = pb_ostream_from_buffer(buf, sizeof(buf));
    if (!pb_encode(&stream, TrackerPose_fields, &msg)) {
        LOG_ERR("pb_encode failed");
        return;
    }
    bt_gatt_notify(NULL, &pose_svc.attrs[2], buf, stream.bytes_written);
}