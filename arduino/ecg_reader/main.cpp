#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UUID
const char* SERVICE_UUID        = "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a";
const char* CHARACTERISTIC_UUID = "7a1e8b7d-9a3e-4657-927b-339adddc2a5b";

BLECharacteristic* ekgChar;
BLEServer* server;
bool deviceConnected = false;

const int ekgPin = A0;     // GPIO36
const int fs = 128;
unsigned long lastMicros = 0;

// BLE Callbacks
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("✅ BLE Connected");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("⚠️ BLE Disconnected");

    // ⬇️ Automatyczne ponowne rozgłaszanie
    BLEDevice::getAdvertising()->start();
    Serial.println("📡 BLE Advertising restarted...");
  }
};


bool isSubscribed() {
  auto* desc = (BLE2902*)ekgChar->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  return desc && desc->getNotifications();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("🔧 Starting...");

  analogReadResolution(12);  // 0–4095

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
  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->start();

  Serial.println("📡 BLE advertising...");
}

void loop() {
  if (micros() - lastMicros >= (1000000UL / fs)) {
    lastMicros = micros();

    int raw = analogRead(ekgPin);       // 0–4095
    int16_t centered = raw - 2048;      // odchylenie od środka

    // Przeskaluj zakładając: pełna skala ADC (4096) = 3.3V,
    // a zakres ±2048 = ±1650 mV → realny EKG to ~ ±2 mV
    float mv = (centered / 2048.0f) * 2.0f;  // finalny sygnał w mV

    Serial.printf("📤 %.3f mV\n", mv);

    if (deviceConnected && isSubscribed()) {
      uint8_t buffer[4];
      memcpy(buffer, &mv, sizeof(float));
      ekgChar->setValue(buffer, 4);
      ekgChar->notify();
    }
  }
}
