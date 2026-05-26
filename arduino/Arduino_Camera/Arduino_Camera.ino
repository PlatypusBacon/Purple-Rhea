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
const char* HostName = "ESP32-CAM";
const char* mqttUser = "47484333";
const char* mqttPassword = "47484333";
const char* topic_PHOTO = "SMILE";
const char* topic_PUBLISH = "PICTURE";
const char* topic_FLASH = "FLASH";
const int MAX_PAYLOAD = 60000;

bool flash = true;

WiFiClient espClient;
PubSubClient client(espClient);

void startCameraServer();
void setupLedFlash();

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
                    }
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
    uint8_t pose_buf[TrackerPose_size];
    pb_ostream_t ps = pb_ostream_from_buffer(pose_buf, sizeof(pose_buf));
    if (!pb_encode(&ps, TrackerPose_fields, &latest_pose)) {
        return;
    }
    uint16_t pose_len = (uint16_t)ps.bytes_written;

    if (flash) digitalWrite(LED_GPIO_NUM, HIGH);
    camera_fb_t *fb = esp_camera_fb_get();
    digitalWrite(LED_GPIO_NUM, LOW);
    if (!fb) return;

    uint32_t total = 2 + pose_len + fb->len;
    if (total > MAX_PAYLOAD) {
        esp_camera_fb_return(fb);
        return;
    }

    uint8_t *pkt = (uint8_t *)malloc(total);
    if (!pkt) {
        esp_camera_fb_return(fb);
        return;
    }

    pkt[0] = pose_len & 0xFF;
    pkt[1] = (pose_len >> 8) & 0xFF;
    memcpy(pkt + 2, pose_buf, pose_len);
    memcpy(pkt + 2 + pose_len, fb->buf, fb->len);
    esp_camera_fb_return(fb);

    client.publish(topic_PUBLISH, pkt, total, false);
    free(pkt);
}

void set_flash() {
  flash = !flash;
}

void callback(String topic, byte* message, unsigned int length) {
  if (topic == topic_PHOTO) { take_picture(); }
  if (topic == topic_FLASH) { set_flash(); }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("MQTT connecting...");
    if (client.connect(HostName, mqttUser, mqttPassword)) {
      Serial.println("connected");
      client.subscribe(topic_PHOTO);
      client.subscribe(topic_FLASH);
    } else {
      Serial.printf("failed rc=%d\n", client.state());
      delay(5000);
    }
  }
}

void setup() {
  // TX-only Serial on GPIO 1 for debug — RX pin disabled (-1) so GPIO 3 stays free
  Serial.begin(115200, SERIAL_8N1, -1, 1);
  Serial.println();

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

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    return;
  }
  Serial.println("Camera init OK");

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
  Serial.println(" connected");

  startCameraServer();
  Serial.printf("Camera Ready! http://%s\n", WiFi.localIP().toString().c_str());

  client.setServer(mqttServer, 1883);
  client.setBufferSize(MAX_PAYLOAD);
  client.setCallback(callback);
}

void loop() {
    if (!client.connected()) reconnect();
    try_read_pose();   // drain UART each loop
    client.loop();
    delay(10);
}