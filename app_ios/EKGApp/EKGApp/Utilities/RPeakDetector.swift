import Accelerate

struct RPeakDetector {
    static func detect(in signal: [Float], fs: Int) -> [Int] {
        let n = signal.count
        // 1) Derivative
        var deriv = [Float](repeating: 0, count: n)
        for i in 1..<n { deriv[i] = signal[i] - signal[i-1] }
        // 2) Squaring
        let squared = deriv.map { $0 * $0 }
        // 3) Moving average (150 ms window)
        let win = Int(0.150 * Float(fs))
        var integrated = [Float](repeating: 0, count: n)
        let kernel = [Float](repeating: 1/Float(win), count: win)
        vDSP_conv(squared, 1, kernel, 1,
                  &integrated, 1,
                  vDSP_Length(n), vDSP_Length(win))
        // 4) Threshold & peak search
        let thresh = (integrated.max() ?? 0) * 0.5
        let minDist = Int(0.200 * Float(fs))
        var peaks = [Int]()
        var i = win
        while i < n - win {
            if integrated[i] > thresh &&
               integrated[i] > integrated[i-1] &&
               integrated[i] > integrated[i+1] {
                peaks.append(i)
                i += minDist
            } else {
                i += 1
            }
        }
        return peaks
    }
}

struct ECGSegmenter {
    static func segments(from signal: [Float], peaks: [Int], fs: Int = 360) -> [[Float]] {
        let window = fs
        let half = window / 2
        return peaks.map { p in
            let start = max(0, p - half)
            let end   = min(signal.count, p + half)
            var seg = [Float](repeating: 0, count: window)
            let len = end - start
            let offset = half - (p - start)
            seg[offset..<offset+len] = signal[start..<end]
            return seg
        }
    }
}

