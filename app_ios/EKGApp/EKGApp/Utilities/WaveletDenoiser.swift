import Foundation

/// Port of the rhythm-model training denoiser (`wavelet_denoise` in main_model_rythm/single_class/main_single.py):
/// PyWavelets `wavedec` / `waverec` with wavelet 'bior2.6', mode 'symmetric', soft thresholding
/// of the detail levels from `startLevel` on, threshold = scale * sigma * sqrt(2 ln n), sigma = median(|c|) / 0.6745.
/// The DWT/IDWT follow PyWavelets' C implementation so results match it numerically.
enum WaveletDenoiser {

    // pywt.Wavelet('bior2.6') filters
    private static let decLo: [Double] = [0.0, -0.006905339660024878, 0.013810679320049757, 0.04695630968816917,
                                          -0.1077232986963881, -0.16987135563661201, 0.4474660099696121, 0.966747552403483,
                                          0.4474660099696121, -0.16987135563661201, -0.1077232986963881, 0.04695630968816917,
                                          0.013810679320049757, -0.006905339660024878]
    private static let decHi: [Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3535533905932738, -0.7071067811865476,
                                          0.3535533905932738, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    private static let recLo: [Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865476,
                                          0.3535533905932738, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    private static let recHi: [Double] = [0.0, 0.006905339660024878, 0.013810679320049757, -0.04695630968816917,
                                          -0.1077232986963881, 0.16987135563661201, 0.4474660099696121, -0.966747552403483,
                                          0.4474660099696121, 0.16987135563661201, -0.1077232986963881, -0.04695630968816917,
                                          0.013810679320049757, 0.006905339660024878]
    private static var filterLength: Int { decLo.count }

    /// Same parameters as in training: start_level = 4, scale = 0.3, soft threshold, level = min(8, max level).
    static func denoise(_ signal: [Float], startLevel: Int = 4, scale: Double = 0.3, maxLevel: Int = 8) -> [Float] {
        let n = signal.count
        guard n > 0 else { return [] }
        let level = min(maxLevel, dwtMaxLevel(dataLength: n, filterLength: filterLength))
        guard level > 0 else { return signal }

        // wavedec → [cA_level, cD_level, cD_level-1, ..., cD_1]
        var coeffs: [[Double]] = []
        var approx = signal.map(Double.init)
        var details: [[Double]] = []
        for _ in 0..<level {
            let (a, d) = dwt(approx)
            details.append(d)
            approx = a
        }
        coeffs.append(approx)
        coeffs.append(contentsOf: details.reversed())

        // Soft-threshold detail coefficients from startLevel on (indices as in the Python code)
        if startLevel < coeffs.count {
            for i in startLevel..<coeffs.count {
                let c = coeffs[i]
                guard !c.isEmpty, c.allSatisfy({ $0.isFinite }) else { continue }
                let sigma = median(c.map { abs($0) }) / 0.6745
                guard sigma.isFinite, sigma > 1e-6 else { continue }
                let threshold = scale * sigma * (2 * log(Double(c.count))).squareRoot()
                coeffs[i] = c.map { v in
                    let m = abs(v) - threshold
                    return m > 0 ? (v < 0 ? -m : m) : 0
                }
            }
        }

        // waverec
        var a = coeffs[0]
        for d in coeffs.dropFirst() {
            if a.count == d.count + 1 { a.removeLast() }
            a = idwt(a, d)
        }
        return a.prefix(n).map(Float.init)
    }

    // MARK: - PyWavelets-compatible primitives

    static func dwtMaxLevel(dataLength: Int, filterLength: Int) -> Int {
        guard filterLength > 1, dataLength >= filterLength - 1 else { return 0 }
        return Int(floor(log2(Double(dataLength) / Double(filterLength - 1))))
    }

    /// Single-level DWT, mode 'symmetric'. Output length = floor((N + F - 1) / 2).
    static func dwt(_ x: [Double]) -> (approx: [Double], detail: [Double]) {
        let n = x.count, f = filterLength
        let outLen = (n + f - 1) / 2
        var a = [Double](repeating: 0, count: outLen)
        var d = [Double](repeating: 0, count: outLen)
        for o in 0..<outLen {
            let i = 2 * o + 1
            var sa = 0.0, sd = 0.0
            for j in 0..<f {
                let v = x[symmetricIndex(i - j, n)]
                sa += decLo[j] * v
                sd += decHi[j] * v
            }
            a[o] = sa
            d[o] = sd
        }
        return (a, d)
    }

    /// Single-level inverse DWT ("valid" upsampling convolution). Output length = 2N - F + 2.
    static func idwt(_ a: [Double], _ d: [Double]) -> [Double] {
        let n = a.count, f2 = filterLength / 2
        guard n >= f2 else { return [] }
        var out = [Double](repeating: 0, count: 2 * (n - f2 + 1))
        var o = 0
        for i in (f2 - 1)..<n {
            var even = 0.0, odd = 0.0
            for j in 0..<f2 {
                even += recLo[2 * j] * a[i - j] + recHi[2 * j] * d[i - j]
                odd += recLo[2 * j + 1] * a[i - j] + recHi[2 * j + 1] * d[i - j]
            }
            out[o] = even
            out[o + 1] = odd
            o += 2
        }
        return out
    }

    /// Half-sample symmetric extension: ... x1 x0 | x0 x1 ... x(n-1) | x(n-1) x(n-2) ...
    private static func symmetricIndex(_ i: Int, _ n: Int) -> Int {
        let period = 2 * n
        var k = i % period
        if k < 0 { k += period }
        return k < n ? k : period - 1 - k
    }

    private static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        let m = s.count / 2
        return s.count % 2 == 0 ? (s[m - 1] + s[m]) / 2 : s[m]
    }
}
