import CoreML

class WaveformClassifier {
    private let model: WaveModel  // Twój model U-Net
    private let settings = AppSettings.shared

    init() {
        let config = MLModelConfiguration()
        self.model = try! WaveModel(configuration: config)
    }

    /// Zwraca klasy (0–3) dla każdej z 2000 próbek
    func predict(input: [Float]) -> [Int] {
        guard input.count == 2000 else {
            print("❌ Błąd: wejście musi mieć dokładnie 2000 próbek")
            return []
        }

        let mlArray = try! MLMultiArray(shape: [1, 1, 2000], dataType: .float32)
        for i in 0..<2000 {
            mlArray[i] = NSNumber(value: input[i])
        }

        guard let output = try? model.prediction(ecg: mlArray).var_464 else {
            print("❌ Błąd predykcji modelu")
            return []
        }

        var predicted = [Int](repeating: 0, count: 2000)
        for t in 0..<2000 {
            var maxVal: Float32 = -Float.greatestFiniteMagnitude
            var maxIdx = 0
            for c in 0..<4 {
                let idx = t * 4 + c  // zakładamy kształt (1, 2000, 4)
                let val = output[idx].floatValue
                if val > maxVal {
                    maxVal = val
                    maxIdx = c
                }
            }
            predicted[t] = maxIdx
        }

        return predicted  // [Int] – klasy dla każdej próbki
    }

    /// (Opcjonalnie) mapa klas na etykiety
    func label(for classIndex: Int) -> String {
        switch classIndex {
        case 0: return "None"
        case 1: return "P"
        case 2: return "QRS"
        case 3: return "T"
        default: return "?"
        }
    }
}

