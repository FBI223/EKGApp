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

            var leadIndex = 0
            var datIndex = 0
            var visitedDATs = Set<String>()

            while datIndex < meta.datFilenames.count {
                let datFilename = meta.datFilenames[datIndex]
                let datURL = dir.appendingPathComponent(datFilename)

                if visitedDATs.contains(datFilename) {
                    datIndex += 1
                    continue
                }
                visitedDATs.insert(datFilename)

                let datData = try Data(contentsOf: datURL)
                let format = meta.format[datIndex]

                if format == "16" || format == "16+" || format == "80" {
                    var signal: [Float] = []
                    let gain = meta.gains[leadIndex]

                    for j in stride(from: 0, to: datData.count, by: 2) {
                        if j + 1 >= datData.count { break }
                        let sample = datData[j..<j+2].withUnsafeBytes { $0.load(as: Int16.self) }
                        signal.append(Float(sample) / gain)
                    }

                    if leadIndex < leadsSignals.count {
                        leadsSignals[leadIndex] = signal
                    }

                    leadIndex += 1
                    datIndex += 1
                } else {
                    print("❌ Nieobsługiwany format \(format)")
                    datIndex += 1
                }
            }

            if leadsSignals.allSatisfy({ $0.isEmpty }) {
                return nil
            }

            // 🕒 Wczytaj start_time i end_time z komentarzy .hea
            let heaText = try String(contentsOf: heaURL)
            let lines = heaText.components(separatedBy: .newlines)
            let isoFormatter = ISO8601DateFormatter()
            var startTime: Date? = nil
            var endTime: Date? = nil

            for line in lines {
                if line.contains("# start_time:") {
                    let raw = line.replacingOccurrences(of: "# start_time:", with: "").trimmingCharacters(in: .whitespaces)
                    if let parsed = isoFormatter.date(from: raw) {
                        startTime = parsed
                    } else {
                        print("⚠️ Błędny format daty start_time: \(raw)")
                    }
                }
                if line.contains("# end_time:") {
                    let raw = line.replacingOccurrences(of: "# end_time:", with: "").trimmingCharacters(in: .whitespaces)
                    if let parsed = isoFormatter.date(from: raw) {
                        endTime = parsed
                    } else {
                        print("⚠️ Błędny format daty end_time: \(raw)")
                    }
                }
            }

            return ECGLoadedMultiLeadData(
                signals: leadsSignals,
                fs: meta.fs,
                leads: meta.leads,
                startTime: startTime,
                endTime: endTime
            )
        } catch {
            print("❌ Błąd podczas wczytywania WFDB: \(error)")
            return nil
        }
    }

        
    static func parseHeaFile(at heaURL: URL) throws -> ECGSignalMetadata {
        let text = try String(contentsOf: heaURL)
        let lines = text.components(separatedBy: .newlines).filter { !$0.isEmpty }

        print("📄 Wczytuję HEA z: \(heaURL.path)")
        print("🧾 Zawartość HEA:\n\(text)")

        guard let firstLine = lines.first else {
            throw NSError(domain: "HEA_PARSE", code: 1, userInfo: [NSLocalizedDescriptionKey: "Brak pierwszej linii w pliku"])
        }

        let headerParts = firstLine.split(separator: " ")
        guard headerParts.count >= 4 else {
            throw NSError(domain: "HEA_PARSE", code: 2, userInfo: [NSLocalizedDescriptionKey: "Zbyt mało danych w nagłówku"])
        }

        let fs = Int(headerParts[2]) ?? 500
        let nSamples = Int(headerParts[3]) ?? 0

        var leads = [String]()
        var gains = [Float]()
        var datFilenames = [String]()
        var formats = [String]()
        var adcZeros = [Int]()
        var adcResolutions = [Int]()
        var initVals = [Int]()
        var bitResolutions = [Int]()

        for line in lines.dropFirst() where !line.trimmingCharacters(in: .whitespaces).hasPrefix("#") {
            let tokens = line.split(separator: " ", omittingEmptySubsequences: true).map { String($0) }

            print("📤 Tokens: \(tokens)")

            guard tokens.count >= 9 else {
                print("⚠️ Pominięto niekompletną linię: \(tokens)")
                continue
            }

            let datFilename = tokens[0]
            let format = tokens[1]
            let gainStr = tokens[2].split(separator: "/").first.map(String.init) ?? "200"
            let gain = Float(gainStr) ?? 200
            let bitRes = Int(tokens[3]) ?? 16
            let zero = Int(tokens[4]) ?? 0
            let initVal = Int(tokens[5]) ?? 0
            let adcRes = Int(tokens[6]) ?? 200
            let lead = tokens[8]

            datFilenames.append(datFilename)
            formats.append(format)
            gains.append(gain)
            bitResolutions.append(bitRes)
            adcZeros.append(zero)
            initVals.append(initVal)
            adcResolutions.append(adcRes)
            leads.append(lead)
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
