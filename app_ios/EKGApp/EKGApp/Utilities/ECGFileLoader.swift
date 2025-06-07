import Foundation


struct ECGLeadInfo {
    let filename: String
    let format: String
    let gain: Float
    let resolution: Int
    let zero: Int
    let initVal: Int
    let checksum: Int
    let blockSize: Int
    let leadName: String
}



struct ECGRecordingSet: Identifiable {
    var id: String { baseName }
    let baseName: String
    let json: URL
    let wfdbDat: URL
    let wfdbHea: URL
}

struct ECGSignalMetadata {
    let fs: Int
    let nSamples: Int
    let leads: [String]
    let gains: [Float]
    let datFilenames: [String]
    let format: [String]

    let adcZeros: [Int]
    let adcResolutions: [Int]
    let initVals: [Int]
    let bitResolutions: [Int]
}



enum ECGFormat {
    case json
    case wfdb  // plik .hea + .dat
}


struct ECGLoadedMultiLeadData {
    let signals: [[Float]]      // [lead][samples]
    let fs: Int
    let leads: [String]
    let startTime: Date?
    let endTime: Date?
}

enum ECGLoader {
    static func loadMultiLead(from fileURL: URL, format: ECGFormat) -> ECGLoadedMultiLeadData? {
        switch format {
        case .json:
            return loadJSONMultiLead(from: fileURL)
        case .wfdb:
            return loadWFDBMultiLead(baseNameURL: fileURL.deletingPathExtension())
        }
    }

    private static func loadJSONMultiLead(from url: URL) -> ECGLoadedMultiLeadData? {
        do {
            let data = try Data(contentsOf: url)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let rawFs = json["fs"] as? Int,
               let rawSignals = json["signals"] as? [[Double]],
               let rawLeads = json["leads"] as? [String] {
                
                let signals = rawSignals.map { $0.map { Float($0) } }
                let iso = ISO8601DateFormatter()
                let start = (json["start_time"] as? String).flatMap(iso.date)
                let end = (json["end_time"] as? String).flatMap(iso.date)
                
                return ECGLoadedMultiLeadData(signals: signals, fs: rawFs, leads: rawLeads, startTime: start, endTime: end)
            }
        } catch {
            print("❌ JSON load error: \(error)")
        }
        return nil
    }

   
    
    
    
    static func loadWFDBMultiLead(baseNameURL: URL) -> ECGLoadedMultiLeadData? {
        let heaURL = baseNameURL.appendingPathExtension("hea")

        do {
            let meta = try parseHeaFile(at: heaURL)
            let dir = baseNameURL.deletingLastPathComponent()
            var leadsSignals = Array(repeating: [Float](), count: meta.leads.count)
                
            var i = 0
            while i < meta.datFilenames.count {
                let datURL = dir.appendingPathComponent(meta.datFilenames[i])
                let datData = try Data(contentsOf: datURL)
                let format = meta.format[i]
                let gain = meta.gains[i]

                var signal: [Float] = []

                if format == "16" || format == "16+" || format.hasPrefix("16") {
                    for j in stride(from: 0, to: datData.count, by: 2) {
                        if j + 1 >= datData.count { break }
                        let sample = datData[j..<j+2].withUnsafeBytes { $0.load(as: Int16.self) }
                        signal.append(Float(sample) / gain)
                    }
                    leadsSignals[i] = signal
                    i += 1

                } else if format == "212" {
                    
                    
                    guard datData.count % 3 == 0 else {
                        print("⚠️ 212 format: Unexpected byte count")
                        i += 2
                        continue
                    }
                    
                    var signal1: [Float] = []
                    var signal2: [Float] = []
                    for j in stride(from: 0, to: datData.count - 2, by: 3) {
                        let byte1 = datData[j]
                        let byte2 = datData[j + 1]
                        let byte3 = datData[j + 2]

                        let s1 = Int16(Int(byte1) | ((Int(byte3 & 0x0F)) << 8))
                        let s2 = Int16(Int(byte2) | ((Int(byte3 & 0xF0)) << 4))

                        signal1.append(Float(s1) / gain)
                        signal2.append(Float(s2) / gain)
                    }

                    if i < leadsSignals.count { leadsSignals[i] = signal1 }
                    if i + 1 < leadsSignals.count { leadsSignals[i + 1] = signal2 }
                    i += 2  // 212 contains two channels in one file

                } else {
                    print("❌ Unsupported format \(format)")
                    i += 1
                }
            }


            let start: Date? = nil
            let end: Date? = nil



            return ECGLoadedMultiLeadData(signals: leadsSignals, fs: meta.fs, leads: meta.leads, startTime: start, endTime: end)

        } catch {
            print("❌ WFDB load error: \(error)")
            return nil
        }
    }

    
    
    
    
    static func parseHeaFile(at heaURL: URL) throws -> ECGSignalMetadata {
        let text = try String(contentsOf: heaURL)
        let lines = text.split(separator: "\n").map(String.init)

        guard let firstLine = lines.first else { throw NSError(domain: "HEA_PARSE", code: 1) }
        let parts = firstLine.split(separator: " ")
        guard parts.count >= 4 else { throw NSError(domain: "HEA_PARSE", code: 2) }

        let fs = Int(parts[2]) ?? 500
        let nSamples = Int(parts[3]) ?? 0

        var leads = [String]()
        var gains = [Float]()
        var datFilenames = [String]()
        var formats = [String]()
        var adcZeros = [Int]()
        var adcResolutions = [Int]()
        var initVals = [Int]()
        var bitResolutions = [Int]()

        for line in lines.dropFirst().filter({ !$0.starts(with: "#") }) {
            let tokens = line.split(separator: " ")
            guard tokens.count >= 10 else { continue }

            datFilenames.append(String(tokens[0]))
            formats.append(String(tokens[1]))

            let gain = Float(tokens[2].split(separator: "/").first ?? "200") ?? 200
            gains.append(gain)

            bitResolutions.append(Int(tokens[3]) ?? 16)
            adcZeros.append(Int(tokens[4]) ?? 0)
            initVals.append(Int(tokens[5]) ?? 0)
            adcResolutions.append(Int(tokens[6]) ?? 200)

            leads.append(String(tokens[9]))
        }

        return ECGSignalMetadata(
            fs: fs,
            nSamples: nSamples,
            leads: leads,
            gains: gains,
            datFilenames: datFilenames,
            format: formats,
            adcZeros: adcZeros,
            adcResolutions: adcResolutions,
            initVals: initVals,
            bitResolutions: bitResolutions
        )
    }

    
    
}
