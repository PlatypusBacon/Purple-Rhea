#include <Arduino.h>
#include "esp_camera.h"

// AI Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define SERIAL_BAUD     921600
#define TRIGGER_COMMAND "capture"
#define FLASH_LED_PIN   4   // AI Thinker ESP32-CAM onboard white flash LED

bool camera_ready = false;

void captureAndSend();

void setup() {
  delay(1000);
  Serial.begin(SERIAL_BAUD);
  Serial.setDebugOutput(false);
  Serial.println("STATUS:booting");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0  = Y2_GPIO_NUM;
  config.pin_d1  = Y3_GPIO_NUM;
  config.pin_d2  = Y4_GPIO_NUM;
  config.pin_d3  = Y5_GPIO_NUM;
  config.pin_d4  = Y6_GPIO_NUM;
  config.pin_d5  = Y7_GPIO_NUM;
  config.pin_d6  = Y8_GPIO_NUM;
  config.pin_d7  = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_UXGA;   // 1600 x 1200
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size  = FRAMESIZE_SVGA;    // 800 x 600 (DRAM limit)
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("STATUS:camera_init_failed:0x%x\n", err);
    // Don't reboot — just halt so Python can still connect
    while (true) { delay(1000); }
  }

  // Fix the yellow/green tint on first capture — make sure AWB is on
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_whitebal(s, 1);   // enable AWB
    s->set_awb_gain(s, 1);   // enable AWB gain
    s->set_wb_mode(s, 0);    // 0 = auto
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_brightness(s, 0);
    s->set_saturation(s, 0);
  }

  camera_ready = true;
  Serial.println("STATUS:ready");
  // Keep printing ready every 2s so Python can catch it
  // even if it connects late
}

void loop() {
  // Re-announce ready periodically so Python can always catch it
  static unsigned long last_announce = 0;
  if (camera_ready && millis() - last_announce > 2000) {
    Serial.println("STATUS:ready");
    last_announce = millis();
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == TRIGGER_COMMAND) {
      captureAndSend();
    } else if (cmd.length() > 0) {
      Serial.println("STATUS:unknown_command");
    }
  }
}

void captureAndSend() {
  Serial.println("STATUS:capturing");

  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(150);   // let exposure + AWB settle with the flash on

  // One warmup frame so AWB/AEC has a fresh reading post-flash.
  camera_fb_t *warm = esp_camera_fb_get();
  if (warm) esp_camera_fb_return(warm);

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    digitalWrite(FLASH_LED_PIN, LOW);
    Serial.println("STATUS:capture_failed");
    return;
  }

  Serial.printf("IMAGE:%u\n", fb->len);
  Serial.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  digitalWrite(FLASH_LED_PIN, LOW);
  Serial.println("\nSTATUS:done");
}