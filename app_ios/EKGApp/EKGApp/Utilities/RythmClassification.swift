import CoreML

class RhythmClassifier{
    private let model: RythmClassifier
    private let settings = AppSettings.shared

    static let windowSeconds: Float = 10
    static let modelFs: Float = 500
    static let modelLength = 5000

    let classNames = [
        "NSR", "AF_FLUTTER", "PAC", "PVC",
        "BBB", "SVT", "AV_BLOCK", "TORSADES"
    ]

    init() {
        let config = MLModelConfiguration()
        self.model = try! RythmClassifier(configuration: config)
    }

    /// `ecgInput` should hold 10 s of signal at `settings.sampleRateIn`.
    func predict(ecgInput: [Float], age: Float, sex: Float) -> String {
        let inputHz = Float(settings.sampleRateIn)

        // Resample to 500 Hz → 5000 samples, then z-score (as in training)
        let resampled = Resampler.resample(
            input: ecgInput,
            srcRate: inputHz,
            dstRate: Self.modelFs,
            outputCount: Self.modelLength
        )
        let normalized = SignalNormalizer.zScore(resampled)

        let ecgArray = try! MLMultiArray(shape: [1, 1, NSNumber(value: Self.modelLength)], dataType: .float32)
        let demoArray = try! MLMultiArray(shape: [1, 2], dataType: .float32)

        for i in 0..<Self.modelLength {
            ecgArray[i] = NSNumber(value: normalized[i])
        }

        // Model divides age by 110 internally
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
}
