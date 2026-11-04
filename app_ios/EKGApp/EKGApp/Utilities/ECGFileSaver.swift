import Foundation

struct ECGFileSaver {
    
    static func saveAll(baseName: String, buffer: [Float], fs: Int, gain: Float, leadName: String, start: Date, end: Date) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try saveDat(to: dir.appendingPathComponent("\(baseName).dat"), buffer: buffer, gain: gain)
        try saveHea(to: dir.appendingPathComponent("\(baseName).hea"), baseName: baseName, buffer: buffer, fs: fs, gain: gain, leadName: leadName, start: start, end: end)
        try saveJSON(to: dir.appendingPathComponent("\(baseName).json"), buffer: buffer, fs: fs, leadName: leadName, start: start, end: end)
    }
    
    /// Converts mV samples to the digital (ADC) values stored in the .dat file.
    /// Int16(_:) traps on NaN/inf or out-of-range values, so sanitize and clamp first.
    static func digitize(_ buffer: [Float], gain: Float) -> [Int16] {
        buffer.map { value in
            let scaled = value * gain
            guard scaled.isFinite else { return 0 }
            let clamped = min(max(scaled.rounded(), Float(Int16.min)), Float(Int16.max))
            return Int16(clamped)
        }
    }

    /// WFDB checksum: 16-bit signed sum of all samples of the signal.
    static func wfdbChecksum(_ samples: [Int16]) -> Int16 {
        var sum: Int16 = 0
        for s in samples { sum &+= s }   // wrapping addition = sum modulo 2^16
        return sum
    }

    static func saveDat(to url: URL, buffer: [Float], gain: Float) throws {
        let int16Signal = digitize(buffer, gain: gain)
        var datData = Data()
        for sample in int16Signal {
            var le = sample.littleEndian
            datData.append(Data(bytes: &le, count: 2))
        }
        try datData.write(to: url)
        print("✅ DAT saved to \(url)")
    }
    
    
    static func saveHea(to url: URL, baseName: String, buffer: [Float], fs: Int, gain: Float, leadName: String, start: Date, end: Date) throws {
        let settings = AppSettings.shared
        let nSamples = buffer.count
        let sex = settings.userSex == 0 ? "M" : "F"
        let age = settings.userAge

        // Must match exactly what saveDat writes, so initVal and checksum are consistent with the .dat file
        let digital = digitize(buffer, gain: gain)
        let adcResolution = 16                        // bits
        let adcZero = 0
        let initVal = Int(digital.first ?? 0)         // value of the first sample
        let checksum = Int(wfdbChecksum(digital))
        let blockSize = 0
        let gainInt = Int(gain)

        let startStr = ISO8601DateFormatter().string(from: start)
        let endStr = ISO8601DateFormatter().string(from: end)
        let durationSec = Int(Double(nSamples) / Double(fs))

        let mainLine = "\(baseName) 1 \(fs) \(nSamples)"

        // WFDB signal line: file format gain adcres adczero initval checksum blocksize description
        let signalLine = "\(baseName).dat 16 \(gainInt) \(adcResolution) \(adcZero) \(initVal) \(checksum) \(blockSize) \(leadName)"

        let comments = [
            "# age: \(age)",
            "# sex: \(sex)",
            "# duration: \(durationSec) seconds",
            "# start_time: \(startStr)",
            "# end_time: \(endStr)",
            "# Recorded via ECG mobile app"
        ]

        let content = ([mainLine, signalLine] + comments).joined(separator: "\n")
        try content.write(to: url, atomically: true, encoding: .utf8)
        print("✅ HEA saved to \(url.lastPathComponent)")
    }

    static func saveJSON(to url: URL, buffer: [Float], fs: Int, leadName: String, start: Date, end: Date) throws {
        let json: [String: Any] = [
            "fs": fs,
            "leads": [leadName],
            "start_time": ISO8601DateFormatter().string(from: start),
            "end_time": ISO8601DateFormatter().string(from: end),
            "signals": [buffer]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
        try data.write(to: url)
        print("✅ JSON saved to \(url)")
    }
}
