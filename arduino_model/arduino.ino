#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "ekg802_5min_signal.h"
#include "ekg820_5min_signal.h"
#include "ekg823_5min_signal.h"

#define BTN_NEXT_RECORD 0  // Przycisk D0

const char* SERVICE_UUID        = "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a";
const char* CHARACTERISTIC_UUID = "7a1e8b7d-9a3e-4657-927b-339adddc2a5b";

BLECharacteristic* ekgChar;
BLEServer* server;
bool deviceConnected = false;

// --- Rekordy
const int16_t* records[] = {
  ekg802_5min,
  ekg820_5min,
  ekg823_5min
};

size_t record_lengths[] = {
  ekg802_5min_len,
  ekg820_5min_len,
  ekg823_5min_len
};

int record_gains[] = {
  EKG802_GAIN,
  EKG820_GAIN,
  EKG823_GAIN
};

int record_bases[] = {
  EKG802_BASELINE,
  EKG820_BASELINE,
  EKG823_BASELINE
};

int record_fs[] = {
  EKG802_FS,
  EKG820_FS,
  EKG823_FS
};

const char* record_names[] = {"802", "820", "823"};

int current_record = 0;
const int total_records = 3;

const int16_t* current_data = nullptr;
size_t current_len = 0;
int current_gain = 1;
int current_baseline = 0;
int current_fs = 250;

size_t pos = 0;
unsigned long last = 0;

bool was_pressed = false;

bool isSubscribed() {
  BLE2902* desc = (BLE2902*)ekgChar->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  return desc && desc->getNotifications();
}

void load_record(int index) {
  current_data = records[index];
  current_len = record_lengths[index];
  current_gain = record_gains[index];
  current_baseline = record_bases[index];
  current_fs = record_fs[index];
  //pos = 0; nie zeruejmy pozycji po przelaczeniu na inny rekord
  Serial.print("✅ Załadowano rekord: ");
  Serial.println(record_names[index]);
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("🔗 BLE klient połączony");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("❌ BLE klient rozłączony");
    delay(100);
    BLEDevice::startAdvertising();
    Serial.println("📡 Reklamowanie ponowione");
  }
};

void setup() {
  Serial.begin(115200);
  pinMode(BTN_NEXT_RECORD, INPUT_PULLUP);

  BLEDevice::init("ESP32_EKG");
  server = BLEDevice::createServer();
  server->setCallbacks(new MyServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  ekgChar = service->createCharacteristic(CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  ekgChar->addDescriptor(new BLE2902());

  service->start();
  BLEDevice::startAdvertising();
  Serial.println("🔋 BLE gotowe. Czekam na połączenie...");

  load_record(current_record);
}

void loop() {
  bool pressed = (digitalRead(BTN_NEXT_RECORD) == LOW);

  if (pressed && !was_pressed) {
    current_record = (current_record + 1) % total_records;
    load_record(current_record);
    delay(300);  // debounce
  }

  was_pressed = pressed;

  if (!current_data) return;

  if (micros() - last >= 1000000UL / current_fs) {
    last = micros();

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
