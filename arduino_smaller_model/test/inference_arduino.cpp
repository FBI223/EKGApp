#include <TensorFlowLite_ESP32.h>
#include "model.h"
#include "class_N.h"
#include "class_S.h"
#include "class_V.h"
#include "class_F.h"
#include "class_Q.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const float* test_classes[5] = {
  (float*)class_N_samples,
  (float*)class_S_samples,
  (float*)class_V_samples,
  (float*)class_F_samples,
  (float*)class_Q_samples
};

const char* class_names[5] = {"N", "S", "V", "F", "Q"};

void setup() {
  Serial.begin(115200);
  delay(1000);

  const tflite::Model* model_ptr = tflite::GetModel(model_tflite);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Zła wersja modelu");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interpreter(model_ptr, resolver, tensor_arena, kTensorArenaSize);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  for (int cls = 0; cls < 5; cls++) {
    Serial.printf("▶️ Klasa rzeczywista: %s\n", class_names[cls]);
    const float* segments = test_classes[cls];

    for (int s = 0; s < 50; s++) {
      for (int i = 0; i < 187; i++) {
        input->data.f[i] = segments[s * 187 + i];
      }

      if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("❌ Błąd predykcji");
        continue;
      }

      int pred = -1;
      float max_val = -1;
      for (int i = 0; i < output->dims->data[1]; i++) {
        float val = output->data.f[i];
        if (val > max_val) {
          max_val = val;
          pred = i;
        }
      }

      Serial.printf("  Pred: %d (%.4f)\n", pred, max_val);
    }
  }
}

void loop() {
  delay(10000);
}
