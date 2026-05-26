#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "tracker_pose.pb.h"
#include <pb_encode.h>
#include <pb_decode.h>

#include "board_config.h"

// WiFi credentials
const char *ssid = "Jgggggg";
const char *password = "Jg200311";

// MQTT config
const char* mqttServer = "10.134.99.217";
const char* HostName = "ESP32-CAM";
const char* mqttUser = "47484333";
const char* mqttPassword = "47484333";
const char* topic_PHOTO = "SMILE";
const char* topic_PUBLISH = "PICTURE";
const char* topic_FLASH = "FLASH";
const int MAX_PAYLOAD = 60000;

bool flash = true;

// ── Tracker UART ──────────────────────────────────────────────────────────────
// Adjust RX/TX pin numbers to match your wiring to the Zephyr tracker board.
// Common choices on AI-Thinker ESP32-CAM: RX=14, TX=15  or  RX=13, TX=12
#define TRACKER_RX_PIN  14
#define TRACKER_TX_PIN  15
#define TRACKER_BAUD    115200

HardwareSerial TrackerSerial(1);        // UART1

// Latest decoded pose — updated every loop() iteration
static TrackerPose latest_pose = TrackerPose_init_zero;
// ─────────────────────────────────────────────────────────────────────────────

WiFiClient espClient;
PubSubClient client(espClient);

void startCameraServer();
void setupLedFlash();

void sendMQTT(const uint8_t* buf, uint32_t len) {
    Serial.println("Sending picture...");
    if (len > MAX_PAYLOAD) {
        Serial.println("Picture too large, increase MAX_PAYLOAD");
    } else {
        Serial.print("Picture sent?: ");
        Serial.println(client.publish(topic_PUBLISH, buf, len, false));
    }
}

// Read one framed protobuf pose message from TrackerSerial.
// Frame format: [0xAA][len_hi][len_lo][payload...][0x55]
bool try_read_pose() {
    if (!TrackerSerial.available()) return false;
    if (TrackerSerial.read() != 0xAA) return false;

    uint16_t len = ((uint16_t)TrackerSerial.read() << 8)
                 |  (uint16_t)TrackerSerial.read();
    if (len == 0 || len > TrackerPose_size) return false;

    uint8_t buf[TrackerPose_size];
    if (TrackerSerial.readBytes(buf, len) < len) return false;
    if (TrackerSerial.read() != 0x55) return false;

    pb_istream_t stream = pb_istream_from_buffer(buf, len);
    return pb_decode(&stream, TrackerPose_fields, &latest_pose);
}

void take_picture() {
    // 1. Encode current pose to temporary buffer
    uint8_t pose_buf[TrackerPose_size];
    pb_ostream_t ps = pb_ostream_from_buffer(pose_buf, sizeof(pose_buf));
    if (!pb_encode(&ps, TrackerPose_fields, &latest_pose)) {
        Serial.println("pose encode failed");
        return;
    }
    uint16_t pose_len = (uint16_t)ps.bytes_written;

    // 2. Capture JPEG
    if (flash) digitalWrite(LED_GPIO_NUM, HIGH);
    camera_fb_t *fb = esp_camera_fb_get();
    digitalWrite(LED_GPIO_NUM, LOW);
    if (!fb) { Serial.println("Camera capture failed"); return; }

    // 3. Build envelope: [pose_len 2B LE][pose][jpeg]
    uint32_t total = 2 + pose_len + fb->len;
    if (total > MAX_PAYLOAD) {
        Serial.println("Packet too large");
        esp_camera_fb_return(fb);
        return;
    }

    uint8_t *pkt = (uint8_t *)malloc(total);
    if (!pkt) {
        Serial.println("malloc failed");
        esp_camera_fb_return(fb);
        return;
    }

    pkt[0] = pose_len & 0xFF;
    pkt[1] = (pose_len >> 8) & 0xFF;
    memcpy(pkt + 2, pose_buf, pose_len);
    memcpy(pkt + 2 + pose_len, fb->buf, fb->len);
    esp_camera_fb_return(fb);

    Serial.printf("Publishing %u bytes (pose=%u jpeg=%u)\n",
                  total, pose_len, total - 2 - pose_len);
    client.publish(topic_PUBLISH, pkt, total, false);
    free(pkt);
}

void set_flash() {
    flash = !flash;
    Serial.print("Flash set to: ");
    Serial.println(flash);
}

void callback(String topic, byte* message, unsigned int length) {
    Serial.println(topic);
    if (topic == topic_PHOTO) { take_picture(); }
    if (topic == topic_FLASH) { set_flash(); }
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        if (client.connect(HostName, mqttUser, mqttPassword)) {
            Serial.println("connected");
            client.subscribe(topic_PHOTO);
            client.subscribe(topic_FLASH);
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" retrying in 5s");
            delay(5000);
        }
    }
}

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    // Tracker UART
    TrackerSerial.begin(TRACKER_BAUD, SERIAL_8N1, TRACKER_RX_PIN, TRACKER_TX_PIN);
    Serial.printf("Tracker UART on RX=%d TX=%d @ %d\n",
                  TRACKER_RX_PIN, TRACKER_TX_PIN, TRACKER_BAUD);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size   = FRAMESIZE_UXGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count     = 1;

    if (config.pixel_format == PIXFORMAT_JPEG) {
        if (psramFound()) {
            config.jpeg_quality = 10;
            config.fb_count     = 2;
            config.grab_mode    = CAMERA_GRAB_LATEST;
        } else {
            config.frame_size  = FRAMESIZE_SVGA;
            config.fb_location = CAMERA_FB_IN_DRAM;
        }
    }

#if defined(CAMERA_MODEL_ESP_EYE)
    pinMode(13, INPUT_PULLUP);
    pinMode(14, INPUT_PULLUP);
#endif

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    if (config.pixel_format == PIXFORMAT_JPEG) {
        s->set_framesize(s, FRAMESIZE_QVGA);
    }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
    s->set_vflip(s, 1);
#endif

#if defined(LED_GPIO_NUM)
    setupLedFlash();
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);
#endif

    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    Serial.print("WiFi connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");

    startCameraServer();
    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.println("' to connect");

    client.setServer(mqttServer, 1883);
    client.setBufferSize(MAX_PAYLOAD);
    client.setCallback(callback);
}

void loop() {
    if (!client.connected()) reconnect();
    try_read_pose();    // drain UART, update latest_pose each loop
    client.loop();
    delay(10);
}
