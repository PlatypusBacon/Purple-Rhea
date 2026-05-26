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
const char* mqttServer = "10.133.32.146";
const char* mqttServer = "10.133.32.146";
const char* HostName = "ESP32-CAM";
const char* mqttUser = "47484333";
const char* mqttPassword = "47484333";
const char* topic_PHOTO = "SMILE";
const char* topic_PUBLISH = "PICTURE";
const char* topic_FLASH = "FLASH";
const int MAX_PAYLOAD = 250000;

bool flash = true;

// Tracker UART (GPIO 14/15 conflict with SD card on AI-Thinker)
#define TRACKER_RX_PIN  13
#define TRACKER_TX_PIN  -1
#define TRACKER_BAUD    115200

HardwareSerial TrackerSerial(1);        // UART1
static TrackerPose latest_pose = TrackerPose_init_zero;

enum PoseRxState : uint8_t {
    RX_WAIT_SOF = 0,
    RX_WAIT_LEN_HI,
    RX_WAIT_LEN_LO,
    RX_WAIT_PAYLOAD,
    RX_WAIT_EOF,
};

static PoseRxState pose_rx_state = RX_WAIT_SOF;
static uint8_t  pose_rx_buf[TrackerPose_size];
static uint16_t pose_rx_len = 0;
static uint16_t pose_rx_idx = 0;
static uint32_t pose_rx_last_byte_ms = 0;
static const uint32_t POSE_RX_TIMEOUT_MS = 50;
static uint32_t pose_rx_ok_count = 0;
static uint32_t pose_rx_fail_count = 0;
static uint32_t pose_rx_last_ok_ms = 0;
static uint32_t pose_rx_last_log_ms = 0;

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

static void reset_pose_rx() {
    pose_rx_state = RX_WAIT_SOF;
    pose_rx_len = 0;
    pose_rx_idx = 0;
}

bool try_read_pose() {
    bool got_pose = false;

    while (TrackerSerial.available()) {
        int raw = TrackerSerial.read();
        if (raw < 0) {
            break;
        }
        uint8_t b = (uint8_t)raw;

        uint32_t now = millis();
        if (pose_rx_state != RX_WAIT_SOF &&
            (now - pose_rx_last_byte_ms) > POSE_RX_TIMEOUT_MS) {
            Serial.println("[POSE RX] timeout while receiving frame, resetting parser");
            reset_pose_rx();
        }
        pose_rx_last_byte_ms = now;

        switch (pose_rx_state) {
            case RX_WAIT_SOF:
                if (b == 0xAA) {
                    pose_rx_state = RX_WAIT_LEN_HI;
                }
                break;

            case RX_WAIT_LEN_HI:
                pose_rx_len = ((uint16_t)b) << 8;
                pose_rx_state = RX_WAIT_LEN_LO;
                break;

            case RX_WAIT_LEN_LO:
                pose_rx_len |= (uint16_t)b;
                if (pose_rx_len == 0 || pose_rx_len > TrackerPose_size) {
                    Serial.printf("[POSE RX] invalid payload length=%u (max=%u)\n",
                                  pose_rx_len, (unsigned)TrackerPose_size);
                    reset_pose_rx();
                } else {
                    pose_rx_idx = 0;
                    pose_rx_state = RX_WAIT_PAYLOAD;
                }
                break;

            case RX_WAIT_PAYLOAD:
                pose_rx_buf[pose_rx_idx++] = b;
                if (pose_rx_idx >= pose_rx_len) {
                    pose_rx_state = RX_WAIT_EOF;
                }
                break;

            case RX_WAIT_EOF:
                if (b == 0x55) {
                    pb_istream_t stream = pb_istream_from_buffer(pose_rx_buf, pose_rx_len);
                    if (pb_decode(&stream, TrackerPose_fields, &latest_pose)) {
                        got_pose = true;
                        pose_rx_ok_count++;
                        pose_rx_last_ok_ms = now;
                        if (pose_rx_ok_count <= 5 || (now - pose_rx_last_log_ms) >= 1000) {
                            Serial.printf("[POSE RX] ok #%lu frame=%lu yaw=%.2f pitch=%.2f roll=%.2f radius=%.3f valid=%d\n",
                                          (unsigned long)pose_rx_ok_count,
                                          (unsigned long)latest_pose.frame_index,
                                          latest_pose.yaw,
                                          latest_pose.pitch,
                                          latest_pose.roll,
                                          latest_pose.radius,
                                          latest_pose.radius_valid ? 1 : 0);
                            pose_rx_last_log_ms = now;
                        }
                    } else {
                        pose_rx_fail_count++;
                        Serial.printf("[POSE RX] decode failed #%lu (len=%u): %s\n",
                                      (unsigned long)pose_rx_fail_count,
                                      pose_rx_len,
                                      PB_GET_ERROR(&stream));
                    }
                } else {
                    Serial.printf("[POSE RX] bad EOF byte 0x%02X (expected 0x55)\n", b);
                }
                if (b == 0xAA) {
                    pose_rx_state = RX_WAIT_LEN_HI;
                    pose_rx_len = 0;
                    pose_rx_idx = 0;
                } else {
                    reset_pose_rx();
                }
                break;
        }
    }

    return got_pose;
}

void take_picture() {
    Serial.printf("[POSE TX] frame=%lu yaw=%.2f pitch=%.2f roll=%.2f\n",
                  (unsigned long)latest_pose.frame_index,
                  latest_pose.yaw,
                  latest_pose.pitch,
                  latest_pose.roll);

    // 1. Encode current pose to temporary buffer
    uint8_t pose_buf[TrackerPose_size];
    pb_ostream_t ps = pb_ostream_from_buffer(pose_buf, sizeof(pose_buf));
    if (!pb_encode(&ps, TrackerPose_fields, &latest_pose)) {
        Serial.printf("pose encode failed: %s\n", PB_GET_ERROR(&ps));
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

void callback(char* topic, byte* message, unsigned int length) {
  String topic_s = String(topic);
  String payload_s;
  payload_s.reserve(length);
  for (unsigned int i = 0; i < length; i++) {
      payload_s += (char)message[i];
  }

  Serial.printf("[MQTT RX] topic=%s payload_len=%u payload='%s'\n",
                topic_s.c_str(), length, payload_s.c_str());

  if (topic_s == topic_PHOTO) { take_picture(); }
  if (topic_s == topic_FLASH) { set_flash(); }
}

void reconnect() {
  while (!client.connected()) {
    Serial.printf("Attempting MQTT connection to %s:%d as %s...\n",
                  mqttServer, 1883, HostName);
    if (client.connect(HostName, mqttUser, mqttPassword)) {
      Serial.println("connected");
      bool photo_ok = client.subscribe(topic_PHOTO);
      bool flash_ok = client.subscribe(topic_FLASH);
      Serial.printf("[MQTT RX] subscribed SMILE=%d FLASH=%d\n",
                    photo_ok ? 1 : 0, flash_ok ? 1 : 0);
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
  TrackerSerial.begin(TRACKER_BAUD, SERIAL_8N1, TRACKER_RX_PIN, TRACKER_TX_PIN);
  Serial.printf("Tracker UART on RX=%d TX=%d @ %d\n",
                TRACKER_RX_PIN, TRACKER_TX_PIN, TRACKER_BAUD);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      config.frame_size = FRAMESIZE_SVGA;
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

static uint32_t uart_debug_timer = 0;
static uint32_t uart_bytes_seen = 0;

void loop() {
    if (!client.connected()) reconnect();
    bool got_pose = try_read_pose();   // drain UART each loop
    if (!got_pose && pose_rx_last_ok_ms > 0 && (millis() - pose_rx_last_ok_ms) > 3000) {
        Serial.println("[POSE RX] no valid tracker pose for >3s");
        pose_rx_last_ok_ms = millis();
    }

    int avail = TrackerSerial.available();
    if (avail > 0) uart_bytes_seen += avail;

    if (millis() - uart_debug_timer > 2000) {
        Serial.printf("[UART debug] bytes_seen=%u  available=%d\n",
                      uart_bytes_seen, TrackerSerial.available());
        uart_debug_timer = millis();
    }

    try_read_pose();
    client.loop();
    delay(10);
}
