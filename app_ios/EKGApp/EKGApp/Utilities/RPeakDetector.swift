import Foundation
import Accelerate

struct RPeakDetector {

    /// Pan–Tompkins–style detector. Returns sample indices of R peaks in `signal`.
    static func detectRPeaks(signal: [Float], fs: Int, sensitivity: Float = 0.6) -> [Int] {
        // Need at least a few samples for the 5-point derivative and a valid sample rate
        guard fs > 0, signal.count >= 5 else { return [] }

        // 1. Derivative → squaring → centred moving-window integration (150 ms)
        let diff = differentiate(signal: signal)
        let squared = diff.map { $0 * $0 }
        let windowSize = max(1, Int(0.150 * Float(fs)))
        let integrated = movingAverageCentered(signal: squared, windowSize: windowSize)

        // 2. Thresholding
        guard let maxValue = integrated.max(), maxValue > 0 else { return [] }
        let threshold = maxValue * sensitivity

        // 3. Local maxima of the integrated signal, with ~200 ms refractory period
        let minDistance = max(1, Int(0.2 * Float(fs)))
        let candidates = findPeaks(signal: integrated, threshold: threshold, minDistance: minDistance)

        // 4. Refine each candidate to the actual R peak in the original signal
        //    (largest deviation from the local baseline within ±search window)
        let searchRadius = max(1, Int(0.075 * Float(fs)))
        var peaks: [Int] = []
        for c in candidates {
            let r = refine(signal: signal, around: c, radius: searchRadius)
            if let last = peaks.last, r - last < minDistance { continue }
            peaks.append(r)
        }
        return peaks
    }

    private static func differentiate(signal: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: signal.count)
        guard signal.count >= 5 else { return output }
        for i in 2..<(signal.count - 2) {
            output[i] = (2 * signal[i + 1] + signal[i + 2] - signal[i - 2] - 2 * signal[i - 1]) / 8.0
        }
        return output
    }

    /// Centred moving average with the same length as the input.
    /// The input is zero-padded so vDSP_conv never reads past the end of the buffer
    /// (vDSP_conv requires `outputCount + filterLength - 1` input samples).
    private static func movingAverageCentered(signal: [Float], windowSize: Int) -> [Float] {
        let n = signal.count
        guard n > 0, windowSize > 1 else { return signal }

        let padLeft = windowSize / 2
        let padRight = windowSize - 1 - padLeft
        var padded = [Float](repeating: 0, count: padLeft)
        padded.reserveCapacity(n + windowSize - 1)
        padded.append(contentsOf: signal)
        padded.append(contentsOf: [Float](repeating: 0, count: padRight))

        var result = [Float](repeating: 0, count: n)
        let kernel = [Float](repeating: 1.0 / Float(windowSize), count: windowSize)
        vDSP_conv(padded, 1, kernel, 1, &result, 1, vDSP_Length(n), vDSP_Length(windowSize))
        return result
    }

    private static func findPeaks(signal: [Float], threshold: Float, minDistance: Int) -> [Int] {
        guard signal.count >= 3 else { return [] }
        var peaks: [Int] = []

        for i in 1..<(signal.count - 1) {
            guard signal[i] > threshold, signal[i] >= signal[i - 1], signal[i] > signal[i + 1] else { continue }
            if let last = peaks.last, i - last < minDistance {
                // Keep the stronger of two close candidates
                if signal[i] > signal[last] { peaks[peaks.count - 1] = i }
                continue
            }
            peaks.append(i)
        }
        return peaks
    }

    private static func refine(signal: [Float], around center: Int, radius: Int) -> Int {
        let lo = max(0, center - radius)
        let hi = min(signal.count - 1, center + radius)
        guard lo < hi else { return center }

        let slice = signal[lo...hi]
        let baseline = slice.reduce(0, +) / Float(slice.count)
        var bestIndex = center
        var bestValue: Float = -1
        for i in lo...hi {
            let v = abs(signal[i] - baseline)
            if v > bestValue {
                bestValue = v
                bestIndex = i
            }
        }
        return bestIndex
    }
}
