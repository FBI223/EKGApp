

import Foundation
import Accelerate

struct RPeakDetector {
    
    static func detectRPeaks(signal: [Float], fs: Int, sensitivity: Float = 0.6 ) -> [Int] {
        guard signal.count > 0 else { return [] }

        
        // 1. Bandpass filtering (5–15 Hz) — FIR using simple difference (approximation)
        let diff = differentiate(signal: signal)
        let squared = diff.map { $0 * $0 }
        let integrated = movingAverage(signal: squared, windowSize: Int(0.150 * Float(fs))) // 150ms

        // 2. Thresholding
        let threshold = integrated.max()! * sensitivity

        // 3. Find local maxima above threshold with min spacing (~200ms refractory)
        let minDistance = Int(0.2 * Float(fs))
        let peaks = findPeaks(signal: integrated, threshold: threshold, minDistance: minDistance)


        
        return peaks
    }

    private static func differentiate(signal: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: signal.count)
        for i in 2..<signal.count - 2 {
            output[i] = (2 * signal[i + 1] + signal[i + 2] - signal[i - 2] - 2 * signal[i - 1]) / 8.0
        }
        return output
    }

    private static func movingAverage(signal: [Float], windowSize: Int) -> [Float] {
        var result = [Float](repeating: 0, count: signal.count)
        let kernel = [Float](repeating: 1.0 / Float(windowSize), count: windowSize)
        vDSP_conv(signal, 1, kernel, 1, &result, 1, vDSP_Length(signal.count), vDSP_Length(windowSize))
        return result
    }

    private static func findPeaks(signal: [Float], threshold: Float, minDistance: Int) -> [Int] {
        var peaks: [Int] = []
        var lastPeakIndex: Int? = nil

        for i in 1..<signal.count - 1 {
            if signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                if let last = lastPeakIndex, i - last < minDistance {
                    continue
                }
                peaks.append(i)
                print("wykryto qrs!")
                lastPeakIndex = i
            }
        }

        return peaks
    }

}


