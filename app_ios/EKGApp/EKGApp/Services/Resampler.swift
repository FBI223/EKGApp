



import SwiftUI
import CoreBluetooth
import Accelerate
import CoreML
import UniformTypeIdentifiers


// MARK: - Resampler

struct Resampler {
    static func resample(input: [Float], srcRate: Float, dstRate: Float) -> [Float] {
        let ratio = dstRate / srcRate
        let outputCount = Int(Float(input.count) * ratio)
        var output = [Float](repeating: 0, count: outputCount)
        for i in 0..<outputCount {
            let srcIndex = Float(i) / ratio
            let lower = Int(floor(srcIndex))
            let upper = min(lower + 1, input.count - 1)
            let t = srcIndex - Float(lower)
            output[i] = input[lower] * (1 - t) + input[upper] * t
        }
        return output
    }
}
