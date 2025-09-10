#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

#include "esp_camera.h"
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#include <ESP32Servo.h>
#define SERVO_SELECT  13
#define SERVO_RELEASE 14

#define FULL_LOG false 

Servo s_sel, s_rel;

#include "wifi_creds.h"


bool waiting_for_response = false;





void sendStatus(char* status) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        String logUrl = String(serverUrl) + String(endpointLog);
        http.begin(logUrl);
        http.addHeader("Content-Type", "application/json");
        String payload = String("{\"status\": \"") + String(status) + String("\"}");
        int res = http.POST(payload);
        // Serial.println("Log response: " + String(res));
        http.end();
    } else {
        Serial.println("WiFi Disconnected!");
    }
}

int awaitResponse() {
    // get response from server
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        String url = String(serverUrl) + String(endpointResponse);
        http.begin(url);
        int httpResponseCode = http.GET();
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println(response);
            if(response.indexOf("result") == -1) {
                http.end();
                return -1;
            }
            Serial.println("response: " + response.substring(response.indexOf("result") + 8, response.indexOf("result") + 9));

            int result = response.substring(response.indexOf("result") + 8, response.indexOf("result") + 9).toInt();
            Serial.println("result: " + String(result));
            
            http.end();
            return result;
        }
    }
    return -1;
}


float select_position = 0;
float target_position = 0;

void smooth_move(float target) {
    target_position = target;
    while (select_position != target_position) {
        if (select_position < target_position) {
            select_position += 1;
        } else {
            select_position -= 1;
        }
        s_sel.write(select_position);
        delay(5);
    }
}


void setup() {
    Serial.begin(115200);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");

    // Initialize camera
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
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera initialization failed");
        return;
    }
    Serial.println("Camera initialized successfully");

    // ajustare senzor
    sensor_t * s = esp_camera_sensor_get();

    s->set_gain_ctrl(s, 0);                       // auto gain off
    s->set_awb_gain(s, 1);                        // Auto White Balance enable (0 or 1)
    // s->set_whitebal(s, 1);       // Enable white balance
    // s->set_wb_mode(s, 3);
    s->set_exposure_ctrl(s, 1);                   // auto exposure off
    s->set_brightness(s, 2);                     // (-2 to 2) - set brightness
    s->set_contrast(s, 2);

    s->set_agc_gain(s, 3);          // set gain manually (0 - 30)
    // s->set_aec_value(s, 800);     // set exposure manually  (0-1200)


    // SERVO
    s_sel.attach(SERVO_SELECT);
    s_rel.attach(SERVO_RELEASE);

    s_sel.write(0);
    s_rel.write(0);


    // PRIMA POZA
    delay(1000);
    // poza initiala pt ajustarea senzorului
    camera_fb_t* fb = esp_camera_fb_get();
    esp_camera_fb_return(fb);
    
    // poza buna
    fb = esp_camera_fb_get();
    
    if (!fb) {
        if(FULL_LOG) sendStatus("initial capture failed");
    }
    else {
        if(FULL_LOG) sendStatus("initial image captured");

        HTTPClient http;
        String initUrl = String(serverUrl) + String(endpointInit);
        http.begin(initUrl);
        http.addHeader("Content-Type", "image/jpeg");

        int res = http.POST(fb->buf, fb->len);

        http.end();
        esp_camera_fb_return(fb);
    }
    
    esp_camera_fb_return(fb);
    delay(1000);
}

void loop() {



    if (WiFi.status() == WL_CONNECTED) {

        
        if(waiting_for_response) {
            int result = awaitResponse();
            if(result == -1) {
                if(FULL_LOG) sendStatus("waiting for reply");
            }
            else {
                if(result == 1) {
                    // s_sel.write(0);
                    smooth_move(0);
                    delay(500);
                    s_rel.write(45);
                    delay(300);
                    s_rel.write(0);
                    
                    if(FULL_LOG) sendStatus("toggle 1");
                }
                else if(result == 2) {
                    // s_sel.write(90);
                    smooth_move(90);
                    delay(500);
                    s_rel.write(45);
                    delay(300);
                    s_rel.write(0);

                    if(FULL_LOG) sendStatus("toggle 2");
                }
                else if(result == 3) {
                    // s_sel.write(180);
                    smooth_move(180);
                    delay(500);
                    s_rel.write(45);
                    delay(300);
                    s_rel.write(0);

                    if(FULL_LOG) sendStatus("toggle 3");
                }
                else {
                    if(FULL_LOG) sendStatus("toggle unknown");
                }
                waiting_for_response = false;
            }
        }

        else {
            // poza
            camera_fb_t* fb = esp_camera_fb_get();
            if (!fb) {
                if(FULL_LOG) sendStatus("capture failed");
                delay(1000);
                return;
            }
            if(FULL_LOG) sendStatus("new image captured");

            HTTPClient http;
            String uploadUrl = String(serverUrl) + String(endpointUpload);
            http.begin(uploadUrl);
            http.addHeader("Content-Type", "image/jpeg");

            int httpResponseCode = http.POST(fb->buf, fb->len);
            if (httpResponseCode > 0) {
                String response = http.getString();
                // Serial.println("Server Response: " + response);

                if (response.indexOf("valid") != -1) {
                    if (response.indexOf("\"valid\":true") != -1) {
                        waiting_for_response = true;
                        if(FULL_LOG) sendStatus("entering wait mode");
                    }
                }
            }

            http.end();
            esp_camera_fb_return(fb);
        }
        

    } else {
        Serial.println("WiFi Disconnected!");
        WiFi.reconnect();
    }

    delay(5000);
}
