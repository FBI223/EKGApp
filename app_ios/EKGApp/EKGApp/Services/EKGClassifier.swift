import CoreML

class EKGClassifier {
    private let model: ECGClassifier   // jeśli plik to ECGClassifier.mlmodel
    init() {
        let config = MLModelConfiguration()
        model = try! ECGClassifier(configuration: config)
    }
    func predict(_ samples: [Float]) -> String {
        // MOCK — zwracaj losową lub stałą klasę
        let mockClasses = ["Normal", "PVC", "AF", "PAC", "Noise"]
        return mockClasses.randomElement() ?? "Unknown"
    }
}
