import Foundation

/// A pre-recorded ECG signal bundled with the app (folder `SimulationSignals/`, files `sim_*.json`),
/// streamed by `EKGBLEManager` as if it came from a real BLE ECG device.
struct SimulatedSignal: Identifiable, Hashable {
    let id: String          // file name without extension, e.g. "sim_802"
    let name: String
    let description: String
    let fs: Int
    let samples: [Float]    // mV

    var durationSeconds: Int { fs > 0 ? samples.count / fs : 0 }

    static func == (lhs: SimulatedSignal, rhs: SimulatedSignal) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

enum SimulatedSignalLibrary {
    private struct FileFormat: Decodable {
        let name: String?
        let description: String?
        let fs: Int
        let signals: [[Float]]
    }

    /// Loads every `sim_*.json` from the app bundle (also accepts a `SimulationSignals` subfolder).
    static func loadAll(bundle: Bundle = .main) -> [SimulatedSignal] {
        var urls = bundle.urls(forResourcesWithExtension: "json", subdirectory: nil) ?? []
        urls += bundle.urls(forResourcesWithExtension: "json", subdirectory: "SimulationSignals") ?? []

        var seen = Set<String>()
        let signals: [SimulatedSignal] = urls.compactMap { url in
            let id = url.deletingPathExtension().lastPathComponent
            guard id.hasPrefix("sim_"), !seen.contains(id) else { return nil }
            seen.insert(id)
            return load(url: url, id: id)
        }
        return signals.sorted { $0.id < $1.id }
    }

    static func load(url: URL, id: String) -> SimulatedSignal? {
        do {
            let data = try Data(contentsOf: url)
            let file = try JSONDecoder().decode(FileFormat.self, from: data)
            guard file.fs > 0, let lead = file.signals.first, !lead.isEmpty else { return nil }
            return SimulatedSignal(
                id: id,
                name: file.name ?? id,
                description: file.description ?? "",
                fs: file.fs,
                samples: lead
            )
        } catch {
            print("❌ Cannot load simulation signal \(url.lastPathComponent): \(error)")
            return nil
        }
    }
}
