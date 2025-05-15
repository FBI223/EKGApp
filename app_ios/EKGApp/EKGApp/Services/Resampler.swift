



import SwiftUI
import CoreBluetooth
import Accelerate
import CoreML
import UniformTypeIdentifiers


import Foundation

struct Resampler {
    static func resample(input: [Float], srcRate: Float, dstRate: Float) -> [Float] {
        guard srcRate > 0, dstRate > 0, !input.isEmpty else { return [] }

        let ratio = dstRate / srcRate
        let outputLength = Int(Float(input.count) * ratio)
        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcIndex = Float(i) / ratio
            let lower = Int(floor(srcIndex))
            let upper = min(lower + 1, input.count - 1)
            let t = srcIndex - Float(lower)

            output[i] = (1 - t) * input[lower] + t * input[upper]
        }

        return output
    }
}
