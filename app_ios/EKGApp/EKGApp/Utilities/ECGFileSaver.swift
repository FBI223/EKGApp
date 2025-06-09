import Foundation

struct ECGFileSaver {
    
    static func saveAll(baseName: String, buffer: [Float], fs: Int, gain: Float, leadName: String, start: Date, end: Date) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try saveDat(to: dir.appendingPathComponent("\(baseName).dat"), buffer: buffer, gain: gain)
        try saveHea(to: dir.appendingPathComponent("\(baseName).hea"), baseName: baseName, buffer: buffer, fs: fs, gain: gain, leadName: leadName, start: start, end: end)
        try saveJSON(to: dir.appendingPathComponent("\(baseName).json"), buffer: buffer, fs: fs, leadName: leadName, start: start, end: end)
    }
    
    static func saveDat(to url: URL, buffer: [Float], gain: Float) throws {
        let int16Signal = buffer.map { Int16($0 * gain) }
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

        let resolution = 16         // ilość bitów
        let adcRes = 200            // ADC resolution
        let adcZero = 0             // zero ADC
        let initVal = 0             // initial value
        let bitResolution = 16      // liczba bitów (powtórzone)
        let gainInt = Int(gain)

        let startStr = ISO8601DateFormatter().string(from: start)
        let endStr = ISO8601DateFormatter().string(from: end)
        let durationSec = Int(Double(nSamples) / Double(fs))

        let mainLine = "\(baseName) 1 \(fs) \(nSamples)"

        // dokładnie 9 pól: filename format gain bitRes zero initVal adcRes leadName
        let signalLine = "\(baseName).dat 16 \(gainInt) \(bitResolution) \(adcZero) \(initVal) \(adcRes) 0 \(leadName)"

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
