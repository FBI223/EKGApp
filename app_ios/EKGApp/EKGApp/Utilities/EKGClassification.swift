import CoreML

class EKGClassifier {
    private let model: ECGClassifier
    private let invClassMap = [0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"]
    private let settings = AppSettings.shared

    init() {
        let config = MLModelConfiguration()
        self.model = try! ECGClassifier(configuration: config)
    }

    func predict(input: [Float]) -> String {
        let resampled = resampleLinear(
            signal: input,
            from: Float(settings.sampleRateIn),
            to: Float(360),
            outputCount: settings.windowLengthBeatResampled
        )

        let mlArray = try! MLMultiArray(shape: [1, 1, NSNumber(value: settings.windowLengthBeatResampled)], dataType: .float32)
        for i in 0..<settings.windowLengthBeatResampled {
            mlArray[i] = NSNumber(value: resampled[i])
        }

        let output = try! model.prediction(input: ECGClassifierInput(input: mlArray)).output
        let maxIdx = (0..<5).max { output[$0].floatValue < output[$1].floatValue }!
        return invClassMap[maxIdx] ?? "?"
    }

    private func resampleLinear(signal: [Float], from srcHz: Float, to dstHz: Float, outputCount: Int) -> [Float] {
        let scale = dstHz / srcHz
        guard signal.count >= 2 else {
            return Array(repeating: signal.first ?? 0.0, count: outputCount)
        }

        var result = [Float](repeating: 0, count: outputCount)
        let maxIndex = signal.count - 1

        for i in 0..<outputCount {
            let index = Float(i) / scale
            let low = Int(floor(index))
            let high = min(low + 1, maxIndex)
            let t = index - Float(low)

            result[i] = (1 - t) * signal[low] + t * signal[high]
        }

        return result
    }
}

