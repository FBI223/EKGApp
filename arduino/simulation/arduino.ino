#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// EKG dane (z 5-minutowych rekordów)
#include "ekg802_5min_signal.h"

// BLE UUID
const char* SERVICE_UUID =        "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a";
const char* CHARACTERISTIC_UUID = "7a1e8b7d-9a3e-4657-927b-339adddc2a5b";

BLECharacteristic* ekgChar;
BLEServer* server;
bool deviceConnected = false;

// Przycisk resetujący (tylko D0)
#define BTN_RESET 0

// Dane EKG
const int16_t* current_data = ekg802_5min;
size_t current_len = ekg802_5min_len;
int current_gain = EKG802_GAIN;
int current_baseline = EKG802_BASELINE;
int current_fs = EKG802_FS;
String current_name = "802";

size_t pos = 0;
unsigned long last = 0;

// === BLE subskrypcja ===
bool isSubscribed() {
  BLE2902* desc = (BLE2902*)ekgChar->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  uint8_t* val = desc ? desc->getValue() : nullptr;
  return val && val[0] == 1;
}

// === BLE Callback ===
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("✅ BLE klient połączony");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("❌ BLE klient rozłączony");
  }
};

// === Setup ===
void setup() {
  Serial.begin(115200);
  Serial.println("🔋 Start ESP32 BLE (bez TFT)");

  pinMode(BTN_RESET, INPUT_PULLUP);

  // BLE init
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
  Serial.println("📡 BLE gotowe, czekam na klienta...");
}

// === Loop ===
void loop() {
  // Przycisk D0 – reset odtwarzania
  if (digitalRead(BTN_RESET) == LOW) {
    pos = 0;
    Serial.println("🔁 Restart od początku (D0)");
    delay(300);  // proste oddebouncowanie
  }

  if (!current_data) return;

  if (micros() - last >= 1e6 / current_fs) {
    last = micros();

    int16_t raw = current_data[pos];
    float mv = (raw - current_baseline) / float(current_gain);

    Serial.printf("rek: %s | pos: %d | mV: %.2f\n", current_name.c_str(), (int)pos, mv);

    if (deviceConnected && isSubscribed()) {
      uint8_t buffer[4];
      memcpy(buffer, &mv, 4);
      ekgChar->setValue(buffer, 4);
      ekgChar->notify();
    }

    pos++;
    if (pos >= current_len) pos = 0;
  }
}
