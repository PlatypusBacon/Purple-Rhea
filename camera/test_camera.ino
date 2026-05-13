#include <Arduino.h>
#include "esp_camera.h"
#include "board_config.h"

#define SERIAL_BAUD     115200
#define TRIGGER_COMMAND "capture"

bool camera_ready = false;

void captureAndSend();

void setup() {
  delay(1000);
  Serial.begin(SERIAL_BAUD);
  Serial.setDebugOutput(false);
  Serial.println("STATUS:booting");

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
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size  = FRAMESIZE_SVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("STATUS:camera_init_failed:0x%x\n", err);
    // Don't reboot — just halt so Python can still connect
    while (true) { delay(1000); }
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

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("STATUS:capture_failed");
    return;
  }

  Serial.printf("IMAGE:%u\n", fb->len);
  Serial.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  Serial.println("\nSTATUS:done");
}