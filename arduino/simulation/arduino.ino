#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Arduino.h>
#include <stdlib.h>

#include "ekg802_5min_signal.h"
#include "ekg820_5min_signal.h"
#include "ekg823_5min_signal.h"

// UUID
const char* SERVICE_UUID = "410de486-ba6e-47a9-85c8-715700eba0fa";
const char* CHARACTERISTIC_UUID = "9e79707a-a0aa-4e79-9009-96d643ef755a";

// BLE
BLEServer* server = nullptr;
BLECharacteristic* ekgChar = nullptr;
bool deviceConnected = false;



// === Dane ===
const int16_t* records[] = { ekg802_5min, ekg820_5min, ekg823_5min };
size_t record_lengths[] = { ekg802_5min_len, ekg820_5min_len, ekg823_5min_len };
int record_gains[] = { EKG802_GAIN, EKG820_GAIN, EKG823_GAIN };
int record_bases[] = { EKG802_BASELINE, EKG820_BASELINE, EKG823_BASELINE };
int record_fs[] = { EKG802_FS, EKG820_FS, EKG823_FS };
const char* record_names[] = { "802", "820", "823" };

int current_record = 0;
const int total_records = 3;

const int16_t* current_data = nullptr;
size_t current_len = 0;
int current_gain = 1;
int current_baseline = 0;
int current_fs = 250;

size_t pos = 0;
unsigned long last_sample_time = 0;

// === CALLBACKI BLE ===
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("🔗 Klient połączony");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("❌ Klient rozłączony");
    delay(100);
    BLEDevice::startAdvertising();
  }
};

// === SUBSKRYPCJA ===
bool isSubscribed() {
  BLE2902* desc = (BLE2902*)ekgChar->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  return desc && desc->getNotifications();
}

// === ZAŁADUJ REKORD ===
void loadRecord(int index) {
  current_record = index;
  current_data = records[index];
  current_len = record_lengths[index];
  current_gain = record_gains[index];
  current_baseline = record_bases[index];
  current_fs = record_fs[index];

  Serial.println("✅ Załadowano rekord: " + String(record_names[index]));
}

// === SETUP ===
void setup() {
  Serial.begin(115200);

  BLEDevice::init("ESP32_EKG_SIMULATOR");
  server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  ekgChar = service->createCharacteristic(CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  ekgChar->addDescriptor(new BLE2902());

  service->start();
  BLEDevice::startAdvertising();

  Serial.println("📡 BLE gotowe. Czekam na połączenie...");

  // Losowy rekord przy każdym uruchomieniu
  randomSeed(esp_random());
  int selected = random(0, total_records);
  loadRecord(selected);
}

// === LOOP ===
void loop() {
  if (!current_data) return;

  unsigned long now = micros();
  if (now - last_sample_time >= 1000000UL / current_fs) {
    last_sample_time = now;

    int16_t raw = current_data[pos];
    float mv = (raw - current_baseline) / float(current_gain);

    if (deviceConnected && isSubscribed()) {
      uint8_t buffer[4];
      memcpy(buffer, &mv, sizeof(float));
      ekgChar->setValue(buffer, 4);
      ekgChar->notify();
    }

    pos++;
    if (pos >= current_len) pos = 0;
  }

}
