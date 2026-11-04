import Foundation

struct Resampler {
    /// Linear resampling; output length = input.count * dstRate / srcRate.
    static func resample(input: [Float], srcRate: Float, dstRate: Float) -> [Float] {
        guard srcRate > 0, dstRate > 0, !input.isEmpty else { return [] }
        let outputLength = Int(Float(input.count) * dstRate / srcRate)
        return resample(input: input, srcRate: srcRate, dstRate: dstRate, outputCount: outputLength)
    }

    /// Linear resampling to an exact number of output samples.
    /// Positions past the end of the input repeat the last sample (never reads out of bounds).
    static func resample(input: [Float], srcRate: Float, dstRate: Float, outputCount: Int) -> [Float] {
        guard outputCount > 0 else { return [] }
        guard srcRate > 0, dstRate > 0, !input.isEmpty else {
            return [Float](repeating: 0, count: outputCount)
        }
        guard input.count >= 2 else {
            return [Float](repeating: input[0], count: outputCount)
        }

        let ratio = dstRate / srcRate
        let maxIndex = input.count - 1
        var output = [Float](repeating: 0, count: outputCount)

        for i in 0..<outputCount {
            let srcIndex = Float(i) / ratio
            let low = min(Int(srcIndex), maxIndex)
            let high = min(low + 1, maxIndex)
            let t = min(max(srcIndex - Float(low), 0), 1)
            output[i] = (1 - t) * input[low] + t * input[high]
        }
        return output
    }
}

/// Normalization matching the preprocessing used when training the models.
enum SignalNormalizer {
    /// (x - mean) / (std + eps) — used by rhythm, wave and beat models in training.
    static func zScore(_ signal: [Float], eps: Float = 1e-8) -> [Float] {
        guard !signal.isEmpty else { return [] }
        let n = Float(signal.count)
        let mean = signal.reduce(0, +) / n
        let variance = signal.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / n
        let std = variance.squareRoot() + eps
        return signal.map { ($0 - mean) / std }
    }
}
