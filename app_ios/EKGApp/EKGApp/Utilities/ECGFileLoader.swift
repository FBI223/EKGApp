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
            print("📄 Wczytuję HEA z: \(heaURL.path)")
            let meta = try parseHeaFile(at: heaURL)
            let dir = baseNameURL.deletingLastPathComponent()
            var leadsSignals = Array(repeating: [Float](), count: meta.leads.count)
            print("✅ Parsowanie HEA OK — \(meta.leads.count) kanały: \(meta.leads)")

            var leadIndex = 0
            var datIndex = 0
            var visitedDATs = Set<String>()

            while datIndex < meta.datFilenames.count {
                let datFilename = meta.datFilenames[datIndex]
                let datURL = dir.appendingPathComponent(datFilename)

                // ❌ Unikaj przetwarzania tego samego pliku dwa razy
                if visitedDATs.contains(datFilename) {
                    print("⚠️ Pominięto powtórzony plik: \(datFilename)")
                    datIndex += 1
                    continue
                }
                visitedDATs.insert(datFilename)

                let datData = try Data(contentsOf: datURL)
                let format = meta.format[datIndex]
                let gain = meta.gains[datIndex]

                print("📦 Przetwarzam \(datFilename), format \(format), gain \(gain)")

                if format == "212" {
                    guard datData.count % 3 == 0 else {
                        print("⚠️ Format 212: nieoczekiwana liczba bajtów (\(datData.count))")
                        print("📦 Długość .dat: \(datData.count) bajtów → \(datData.count / 3) trójek")
                        datIndex += 1
                        leadIndex += 2
                        continue
                    }

                    var signal1: [Float] = []
                    var signal2: [Float] = []

                    for j in stride(from: 0, to: datData.count, by: 3) {
                        let byte0 = datData[j]
                        let byte1 = datData[j + 1]
                        let byte2 = datData[j + 2]

                        // Dekoduj próbkę 1
                        var value1 = Int16(Int(byte0) | ((Int(byte2 & 0x0F)) << 8))
                        if value1 > 2047 { value1 -= 4096 }  // 12-bit signed

                        // Dekoduj próbkę 2
                        var value2 = Int16(Int(byte1) | ((Int(byte2 & 0xF0)) << 4))
                        if value2 > 2047 { value2 -= 4096 }

                        signal1.append(Float(value1) / gain)
                        signal2.append(Float(value2) / gain)
                    }


                    if leadIndex < leadsSignals.count { leadsSignals[leadIndex] = signal1 }
                    if (leadIndex + 1) < leadsSignals.count { leadsSignals[leadIndex + 1] = signal2 }

                    print("📈 Kanały \(leadIndex), \(leadIndex + 1) → \(signal1.count) próbek")

                    leadIndex += 2
                    datIndex += 1
                } else if format.hasPrefix("16") {
                    var signal: [Float] = []
                    for j in stride(from: 0, to: datData.count, by: 2) {
                        if j + 1 >= datData.count { break }
                        let sample = datData[j..<j+2].withUnsafeBytes { $0.load(as: Int16.self) }
                        signal.append(Float(sample) / gain)
                    }

                    if leadIndex < leadsSignals.count {
                        leadsSignals[leadIndex] = signal
                    }
                    print("📈 Kanał \(leadIndex) → \(signal.count) próbek")

                    leadIndex += 1
                    datIndex += 1
                } else {
                    print("❌ Nieobsługiwany format \(format)")
                    datIndex += 1
                }
            }

            if leadsSignals.allSatisfy({ $0.isEmpty }) {
                print("⚠️ Wszystkie sygnały są puste")
                return nil
            }

            return ECGLoadedMultiLeadData(
                signals: leadsSignals,
                fs: meta.fs,
                leads: meta.leads,
                startTime: nil,
                endTime: nil
            )
        } catch {
            print("❌ Błąd ładowania WFDB: \(error)")
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
