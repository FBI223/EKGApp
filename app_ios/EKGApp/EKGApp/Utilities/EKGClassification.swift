import CoreML

class EKGClassifier {
    private let model: ECGClassifier
    private let invClassMap = [0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"]
    private let settings = AppSettings.shared

    init() {
        let config = MLModelConfiguration()
        self.model = try! ECGClassifier(configuration: config)
    }

    /// `input`: one beat segment at 360 Hz, `windowLengthBeatResampled` (540) samples centred on R.
    func predict(input: [Float]) -> String {
        let length = settings.windowLengthBeatResampled
        guard input.count >= length else { return "?" }

        // Training used z-score normalization (global over the dataset); per-segment is the closest equivalent
        let normalized = SignalNormalizer.zScore(Array(input.prefix(length)))

        let mlArray = try! MLMultiArray(shape: [1, 1, NSNumber(value: length)], dataType: .float32)
        for i in 0..<length {
            mlArray[i] = NSNumber(value: normalized[i])
        }

        let output = try! model.prediction(input: ECGClassifierInput(input: mlArray)).output
        let maxIdx = (0..<5).max { output[$0].floatValue < output[$1].floatValue }!
        return invClassMap[maxIdx] ?? "?"
    }
}
