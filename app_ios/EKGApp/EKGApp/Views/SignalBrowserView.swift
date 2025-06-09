import SwiftUI
import Charts
import UniformTypeIdentifiers


struct SignalBrowserView: View {
    @State private var recordings: [ECGRecordingSet] = []
    @State private var showShareSheet = false
    @State private var fileToShareURLs: [URL] = []
    @State private var selectedFile: (url: URL, format: ECGFormat)? = nil
    @State private var showAlert = false
    
    @State private var showImportSheet = false
    @State private var alertMessage: String = ""
    @State private var showDeleteAllAlert = false

    

    var body: some View {
        NavigationView {
            List {
                ForEach(recordings) { rec in
                    HStack(spacing: 16) {
                        
                        
                        Menu {
                            if FileManager.default.fileExists(atPath: rec.json.path) {
                                Button("📂 Open JSON") {
                                    checkFileAndNavigate(rec.json, format: .json)
                                }
                            }
                            if FileManager.default.fileExists(atPath: rec.wfdbHea.path),
                               FileManager.default.fileExists(atPath: rec.wfdbDat.path) {
                                Button("📂 Open WFDB") {
                                    checkFileAndNavigate(rec.wfdbHea, format: .wfdb)
                                }
                            }
                        } label: {
                            Text(rec.baseName)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .font(.headline)
                                .foregroundColor(.primary)
                        }



                        Spacer()

                        Menu {
                            Button("📤 JSON") { shareFiles(urls: [rec.json]) }
                            Button("📤 WFDB (.dat + .hea)") {
                                shareFiles(urls: [rec.wfdbDat, rec.wfdbHea])
                            }
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundColor(.blue)
                        }
                        .buttonStyle(BorderlessButtonStyle())

                    }
                }
                .onDelete(perform: deleteAtOffsets)
            }
            .navigationTitle("Saved Records")
        
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    EditButton()
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack {
                        Button(action: loadRecordings) {
                            Image(systemName: "arrow.clockwise")
                        }
                        Button(action: { showImportSheet = true }) {
                            Image(systemName: "square.and.arrow.down")
                        }
                    }
                }
                
                ToolbarItem(placement: .bottomBar) {
                    Button(role: .destructive) {
                        showDeleteAllAlert = true
                    } label: {
                        Label("Delete ALL", systemImage: "trash")
                    }
                }

                
            }
            .sheet(isPresented: $showImportSheet) {
                SignalImportView { urls in
                    handleImportedFiles(urls)
                    loadRecordings()
                }
            }

            

            // Navigation
            .background(
                NavigationLink(
                    destination: selectedFile.map { SignalPlaybackView(fileURL: $0.url, format: $0.format) },
                    isActive: Binding(
                        get: { selectedFile != nil },
                        set: { if !$0 { selectedFile = nil } }
                    )
                ) {
                    EmptyView()
                }.hidden()
            )


            .onAppear(perform: loadRecordings)

            .sheet(isPresented: $showShareSheet) {
                if !fileToShareURLs.isEmpty {
                    ActivityView(activityItems: fileToShareURLs)
                }
            }
            
            .alert("Błąd importu", isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(alertMessage)
            }
            
            .alert("confirm deletion", isPresented: $showDeleteAllAlert) {
                Button("delete all", role: .destructive) {

                    deleteAllRecordings()
                }
                Button("cancel", role: .cancel) { }
            } message: {
                Text("you sure to **delete all** recordings?")
            }

        }
    }
    
    
    
    private func deleteAllRecordings() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            for file in files {
                let ext = file.pathExtension.lowercased()
                if ["json", "hea", "dat"].contains(ext) {
                    try? FileManager.default.removeItem(at: file)
                    print("🗑️ Usunięto: \(file.lastPathComponent)")
                }
            }
            loadRecordings()
        } catch {
            print("❌ Błąd podczas usuwania wszystkich rekordów: \(error)")
        }
    }


    private func checkFileAndNavigate(_ fileURL: URL, format: ECGFormat) {
        print("🔎 Trying to open: \(fileURL.lastPathComponent) as \(format)")

        if let ecg = ECGLoader.loadMultiLead(from: fileURL, format: format),
           ecg.signals.allSatisfy({ !$0.isEmpty }) {
            print("✅ Loaded. Setting selectedFile...")
            selectedFile = (fileURL, format)
        } else {
            print("❌ Failed to load file")
            showAlert = true
        }
    }


        
    
    private func handleImportedFiles(_ urls: [URL]) {
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        var importMap: [String: [URL]] = [:]

        for url in urls {
            let base = url.deletingPathExtension().lastPathComponent
            importMap[base, default: []].append(url)
        }

        for (base, files) in importMap {
            let jsonURL = docDir.appendingPathComponent("\(base).json")
            let heaURL = docDir.appendingPathComponent("\(base).hea")
            let datURL = docDir.appendingPathComponent("\(base).dat")

            let jsonExists = FileManager.default.fileExists(atPath: jsonURL.path)
            let heaExists = FileManager.default.fileExists(atPath: heaURL.path)
            let datExists = FileManager.default.fileExists(atPath: datURL.path)

            // 🚫 Sprawdź konflikt nazw
            if jsonExists || heaExists || datExists {
                showImportError("⚠️ Rekord o nazwie '\(base)' już istnieje. Import został pominięty.")
                continue
            }

            // Tymczasowe buforowanie plików lokalnie
            var jsonSrc: URL? = nil
            var heaSrc: URL? = nil
            var datSrc: URL? = nil

            for file in files {
                let ext = file.pathExtension.lowercased()
                if ext == "json" { jsonSrc = file }
                if ext == "hea" { heaSrc = file }
                if ext == "dat" { datSrc = file }
            }

            // ⛔️ Walidacja formatu HEA (przed kopiowaniem)
            if let heaFile = heaSrc {
                if let meta = try? ECGLoader.parseHeaFile(at: heaFile),
                   !meta.format.allSatisfy({ ["16", "16+", "80"].contains($0) }) {
                    let disallowed = meta.format.filter { !["16", "16+", "80"].contains($0) }
                    showImportError("❌ Odrzucono '\(base)': niedozwolone formaty w HEA: \(disallowed.joined(separator: ", "))")
                    continue
                }
            }

            // ✅ Skopiuj tylko gdy wszystko OK
            for file in files {
                let dest = docDir.appendingPathComponent(file.lastPathComponent)
                try? FileManager.default.copyItem(at: file, to: dest)
            }

            // 🔄 JSON → HEA + DAT
            if let jsonFile = jsonSrc {
                if let ecg = ECGLoader.loadMultiLead(from: jsonFile, format: .json),
                   let signal = ecg.signals.first, ecg.fs > 0 {
                    do {
                        try ECGFileSaver.saveDat(to: datURL, buffer: signal, gain: 200.0)
                        try ECGFileSaver.saveHea(to: heaURL,
                                                 baseName: base,
                                                 buffer: signal,
                                                 fs: ecg.fs,
                                                 gain: 200.0,
                                                 leadName: ecg.leads.first ?? "II",
                                                 start: ecg.startTime ?? Date(),
                                                 end: ecg.endTime ?? Date())
                        print("✅ From JSON → created .hea and .dat for \(base)")
                    } catch {
                        showImportError("❌ Nie udało się zapisać .hea/.dat dla '\(base)': \(error.localizedDescription)")
                    }
                } else {
                    showImportError("❌ Błąd importu JSON: plik '\(base).json' jest nieprawidłowy lub brakuje danych.")
                }
            }

            // 🔄 HEA + DAT → JSON
            if heaSrc != nil && datSrc != nil && jsonSrc == nil {
                if let ecg = ECGLoader.loadMultiLead(from: heaURL, format: .wfdb),
                   let signal = ecg.signals.first, ecg.fs > 0 {
                    do {
                        try ECGFileSaver.saveJSON(to: jsonURL,
                                                  buffer: signal,
                                                  fs: ecg.fs,
                                                  leadName: ecg.leads.first ?? "II",
                                                  start: ecg.startTime ?? Date(),
                                                  end: ecg.endTime ?? Date())
                        print("✅ From HEA → created .json for \(base)")
                    } catch {
                        showImportError("❌ Nie udało się zapisać .json dla '\(base)': \(error.localizedDescription)")
                    }
                } else {
                    showImportError("❌ Nie udało się wczytać HEA lub brak danych dla '\(base)'")
                }
            }
        }

        loadRecordings()
    }

    

    
    
    private func loadRecordings() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        do {
            let allFiles = try FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: .skipsHiddenFiles
            )

            print("📂 All files in directory:")
            for f in allFiles { print(" - \(f.lastPathComponent)") }

            var baseSet = Set<String>()
            var jsonMap: [String: URL] = [:]
            var heaMap: [String: URL] = [:]
            var datMap: [String: URL] = [:]

            for file in allFiles {
                let ext = file.pathExtension.lowercased()
                let base = file.deletingPathExtension().lastPathComponent

                baseSet.insert(base)
                if ext == "json" {
                    jsonMap[base] = file
                } else if ext == "hea" {
                    heaMap[base] = file
                } else if ext == "dat" {
                    datMap[base] = file
                }
            }

            var sets: [ECGRecordingSet] = []

            for base in baseSet {
                let json = jsonMap[base]
                let hea = heaMap[base]
                let dat = datMap[base]

                // Musi istnieć przynajmniej jeden z formatów: JSON lub HEA+DAT
                if json != nil || (hea != nil && dat != nil) {
                    sets.append(.init(
                        baseName: base,
                        json: json ?? dir.appendingPathComponent("\(base).json"),
                        wfdbDat: dat ?? dir.appendingPathComponent("\(base).dat"),
                        wfdbHea: hea ?? dir.appendingPathComponent("\(base).hea")
                    ))
                }
            }

            recordings = sets.sorted {
                let d1 = (try? $0.wfdbHea.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let d2 = (try? $1.wfdbHea.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return d1 > d2
            }

        } catch {
            print("❌ Failed to read files: \(error)")
        }
    }

    
    
    private func showImportError(_ message: String) {
        alertMessage = message
        showAlert = true
    }


    private func shareFiles(urls: [URL]) {
        if urls.contains(where: { !FileManager.default.fileExists(atPath: $0.path) }) {
            print("❌ One or more files to share not found")
            return
        }
        fileToShareURLs = urls
        showShareSheet = true
    }

    private func deleteRecording(_ rec: ECGRecordingSet) {
        for url in [rec.json, rec.wfdbDat, rec.wfdbHea] {
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    try FileManager.default.removeItem(at: url)
                    print("🗑️ Usunięto: \(url.lastPathComponent)")
                } catch {
                    print("❌ Błąd usuwania \(url.lastPathComponent): \(error)")
                }
            } else {
                print("⚠️ Plik nie istnieje: \(url.lastPathComponent)")
            }
        }
        loadRecordings()
    }


    private func deleteAtOffsets(_ offsets: IndexSet) {
        for index in offsets {
            deleteRecording(recordings[index])
        }
    }
    
    
    
    



    private func loadJSONMultiLead(from url: URL) -> ECGLoadedMultiLeadData? {
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

                return ECGLoadedMultiLeadData(
                    signals: signals,
                    fs: rawFs,
                    leads: rawLeads,
                    startTime: start,
                    endTime: end
                )
            }
        } catch {
            print("❌ JSON multi-lead error: \(error)")
        }
        return nil
    }



    
    
}

struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}








