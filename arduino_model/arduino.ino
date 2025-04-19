#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "ekg802_5min_signal.h"

// BLE UUID
const char* SERVICE_UUID        = "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a";
const char* CHARACTERISTIC_UUID = "7a1e8b7d-9a3e-4657-927b-339adddc2a5b";

BLECharacteristic* ekgChar;
BLEServer* server;
bool deviceConnected = false;

// Przycisk restartu (D0)
#define BTN_RESET 0

// Dane EKG
const int16_t* current_data = ekg802_5min;
size_t current_len          = ekg802_5min_len;
int current_gain            = EKG802_GAIN;
int current_baseline        = EKG802_BASELINE;
int current_fs              = EKG802_FS;

size_t pos = 0;
unsigned long last = 0;

// Sprawdzenie subskrypcji
bool isSubscribed() {
  auto* desc = (BLE2902*)ekgChar->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  return desc && desc->getNotifications();
}

// Callback BLE
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    pos = 0;
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    pos = 0;
  }
};

void setup() {
  pinMode(BTN_RESET, INPUT_PULLUP);

  BLEDevice::init("ESP32_EKG");
  server = BLEDevice::createServer();
  server->setCallbacks(new MyServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  ekgChar = service->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  ekgChar->addDescriptor(new BLE2902());

  service->start();
  BLEDevice::getAdvertising()->start();
}

void loop() {
  if (digitalRead(BTN_RESET) == LOW) {
    pos = 0;
    delay(500);  // debounce
  }

  if (!deviceConnected || !isSubscribed()) return;

  if (micros() - last >= (1000000UL / current_fs)) {
    last = micros();

    int16_t raw = current_data[pos];
    float mv = (raw - current_baseline) / float(current_gain);

    uint8_t buffer[4];
    memcpy(buffer, &mv, sizeof(float));
    ekgChar->setValue(buffer, 4);
    ekgChar->notify();

    pos++;
    if (pos >= current_len) pos = 0;
  }
}
