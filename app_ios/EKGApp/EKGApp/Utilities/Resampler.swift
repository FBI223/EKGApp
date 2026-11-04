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

extension Resampler {
    /// Fourier-domain resampling, equivalent to `scipy.signal.resample(x, outputCount)` for real input
    /// (the method used when training the rhythm model). Assumes the signal is periodic, like scipy.
    /// Uses a direct DFT restricted to the kept frequency bins: O(N·K + M·K), a few ms for 10 s windows.
    static func resampleFFT(input: [Float], outputCount num: Int) -> [Float] {
        let nx = input.count
        guard num > 0 else { return [] }
        guard nx > 1 else { return [Float](repeating: input.first ?? 0, count: num) }
        if nx == num { return input }

        let x = input.map(Double.init)
        let n = min(num, nx)
        let nyq = n / 2 + 1                       // number of bins copied from X to Y (scipy: Y[0:nyq] = X[0:nyq])

        // Forward real DFT of the bins we need: X_k = Σ x[t] e^{-2πi k t / nx}
        var re = [Double](repeating: 0, count: nyq)
        var im = [Double](repeating: 0, count: nyq)
        let cosX = (0..<nx).map { cos(2 * Double.pi * Double($0) / Double(nx)) }
        let sinX = (0..<nx).map { sin(2 * Double.pi * Double($0) / Double(nx)) }
        for k in 0..<nyq {
            var sr = 0.0, si = 0.0, idx = 0
            for t in 0..<nx {
                sr += x[t] * cosX[idx]
                si -= x[t] * sinX[idx]
                idx += k
                if idx >= nx { idx -= nx }
            }
            re[k] = sr
            im[k] = si
        }

        // scipy's Nyquist-bin handling for an even length
        if n % 2 == 0 {
            let factor = num < nx ? 2.0 : 0.5
            re[n / 2] *= factor
            im[n / 2] *= factor
        }

        // Inverse real DFT of length num (numpy irfft semantics), scaled by num / nx
        let cosY = (0..<num).map { cos(2 * Double.pi * Double($0) / Double(num)) }
        let sinY = (0..<num).map { sin(2 * Double.pi * Double($0) / Double(num)) }
        let hasNyquistBin = num % 2 == 0 && num / 2 < nyq
        let lastFull = min(nyq - 1, num % 2 == 0 ? num / 2 - 1 : (num - 1) / 2)
        var y = [Float](repeating: 0, count: num)
        for m in 0..<num {
            var s = re[0]                         // irfft ignores the imaginary part of the DC bin
            if lastFull >= 1 {
                var idx = m % num
                for k in 1...lastFull {
                    s += 2 * (re[k] * cosY[idx] - im[k] * sinY[idx])
                    idx += m
                    if idx >= num { idx -= num }
                }
            }
            if hasNyquistBin {
                s += re[num / 2] * (m % 2 == 0 ? 1 : -1)
            }
            y[m] = Float(s / Double(nx))          // (1/num) from irfft × (num/nx) from scipy
        }
        return y
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
