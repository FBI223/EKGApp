import CoreML

class RhythmClassifier{
    private let model: RythmClassifier
    private let settings = AppSettings.shared

    let classNames = [
        "NSR", "AF_FLUTTER", "PAC", "PVC",
        "BBB", "SVT", "AV_BLOCK", "TORSADES"
    ]

    init() {
        let config = MLModelConfiguration()
        self.model = try! RythmClassifier(configuration: config)
    }

    func predict(ecgInput: [Float], age: Float, sex: Float) -> String {
        let inputHz = Float(settings.sampleRateIn)
        let outputHz: Float = 500.0
        let outputLength = 5000

        // Resample from inputHz (e.g. 128Hz) to 500Hz → 5000 samples
        let resampled = resampleLinear(
            signal: ecgInput,
            from: inputHz,
            to: outputHz,
            outputCount: outputLength
        )

        let ecgArray = try! MLMultiArray(shape: [1, 1, 5000], dataType: .float32)
        let demoArray = try! MLMultiArray(shape: [1, 2], dataType: .float32)

        for i in 0..<5000 {
            ecgArray[i] = NSNumber(value: resampled[i])
        }

        demoArray[0] = NSNumber(value: age)
        demoArray[1] = NSNumber(value: sex)

        let input = RythmClassifierInput(ecg: ecgArray, demo: demoArray)
        let output = try! model.prediction(input: input)

        let logits = output.var_291
        var bestIndex = 0
        var maxValue = logits[0].floatValue

        for i in 1..<classNames.count {
            let value = logits[i].floatValue
            if value > maxValue {
                maxValue = value
                bestIndex = i
            }
        }

        return classNames[bestIndex]
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
