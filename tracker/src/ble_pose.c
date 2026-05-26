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
#define BT_UUID_POSE_REQ_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef2)

static struct bt_uuid_128 pose_svc_uuid = BT_UUID_INIT_128(BT_UUID_POSE_SVC_VAL);
static struct bt_uuid_128 pose_chr_uuid = BT_UUID_INIT_128(BT_UUID_POSE_CHR_VAL);
static struct bt_uuid_128 pose_req_uuid = BT_UUID_INIT_128(BT_UUID_POSE_REQ_VAL);
extern const struct bt_gatt_service_static pose_svc;
static bool notify_enabled = false;

/* Set by processing_thread each sample tick — always fresh */
static float  _roll, _pitch, _yaw, _radius;
static bool   _radius_valid;
static uint32_t _frame_index;

static void ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    LOG_INF("pose notify %s", notify_enabled ? "enabled" : "disabled");
}

/* Pi writes anything to this characteristic → XIAO sends one notification */
static ssize_t on_pose_request(struct bt_conn *conn,
                               const struct bt_gatt_attr *attr,
                               const void *buf, uint16_t len,
                               uint16_t offset, uint8_t flags)
{
    if (!notify_enabled) {
        return len;
    }

    TrackerPose msg = TrackerPose_init_zero;
    msg.roll         = _roll;
    msg.pitch        = _pitch;
    msg.yaw          = _yaw;
    msg.radius       = _radius;
    msg.radius_valid = _radius_valid;
    msg.frame_index  = _frame_index;

    uint8_t encode_buf[TrackerPose_size];
    pb_ostream_t stream = pb_ostream_from_buffer(encode_buf, sizeof(encode_buf));
    if (!pb_encode(&stream, TrackerPose_fields, &msg)) {
        LOG_ERR("pb_encode failed");
        return len;
    }

    /* pose_svc.attrs layout:
     *   [0] primary service
     *   [1] pose notify characteristic declaration
     *   [2] pose notify characteristic value   ← notify from here
     *   [3] CCC descriptor
     *   [4] request write characteristic declaration
     *   [5] request write characteristic value
     */
    bt_gatt_notify(NULL, &pose_svc.attrs[2], encode_buf, stream.bytes_written);
    return len;
}

BT_GATT_SERVICE_DEFINE(pose_svc,
    BT_GATT_PRIMARY_SERVICE(&pose_svc_uuid),

    /* Notify characteristic — Pi subscribes, XIAO pushes on request */
    BT_GATT_CHARACTERISTIC(&pose_chr_uuid.uuid,
                           BT_GATT_CHRC_NOTIFY,
                           BT_GATT_PERM_NONE,
                           NULL, NULL, NULL),
    BT_GATT_CCC(ccc_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),

    /* Request characteristic — Pi writes to trigger a single notify */
    BT_GATT_CHARACTERISTIC(&pose_req_uuid.uuid,
                           BT_GATT_CHRC_WRITE_WITHOUT_RESP,
                           BT_GATT_PERM_WRITE,
                           NULL, on_pose_request, NULL),
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

/* Called from processing_thread every sample — just updates the cached values */
void ble_pose_update(float roll, float pitch, float yaw,
                     float radius, bool radius_valid, uint32_t frame_index)
{
    _roll         = roll;
    _pitch        = pitch;
    _yaw          = yaw;
    _radius       = radius;
    _radius_valid = radius_valid;
    _frame_index  = frame_index;
}