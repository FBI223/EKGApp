import SwiftUI
import Charts
import UniformTypeIdentifiers


struct SignalBrowserView: View {
    @State private var recordings: [ECGRecordingSet] = []
    @State private var showShareSheet = false
    @State private var fileToShareURLs: [URL] = []
    @State private var selectedFile: (url: URL, format: ECGFormat)? = nil
    @State private var showAlert = false
    
    // SignalBrowserView.swift
    @State private var showImportSheet = false

    
    

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

            .alert("Invalid File", isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("This ECG recording is not valid and cannot be viewed.")
            }
        }
    }

    private func checkFileAndNavigate(_ fileURL: URL, format: ECGFormat) {
        print("🔎 Trying to open: \(fileURL.lastPathComponent) as \(format)")

        if let _ = ECGLoader.loadMultiLead(from: fileURL, format: format) {
            print("✅ Loaded. Setting selectedFile...")
            selectedFile = (fileURL, format)
        } else {
            print("❌ Failed to load file")
            showAlert = true
        }
    }


    
    
    private func handleImportedFiles(_ urls: [URL]) {
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        var jsonSet = Set<String>()
        var heaSet = Set<String>()

        for url in urls {
            let base = url.deletingPathExtension().lastPathComponent
            let dest = docDir.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.copyItem(at: url, to: dest)

            if url.pathExtension.lowercased() == "json" {
                jsonSet.insert(base)
            } else if url.pathExtension.lowercased() == "hea" {
                heaSet.insert(base)
            }
        }

        let allBases = jsonSet.union(heaSet)

        for base in allBases {
            let jsonURL = docDir.appendingPathComponent("\(base).json")
            let heaURL = docDir.appendingPathComponent("\(base).hea")
            let datURL = docDir.appendingPathComponent("\(base).dat")

            let jsonExists = FileManager.default.fileExists(atPath: jsonURL.path)
            let heaExists = FileManager.default.fileExists(atPath: heaURL.path)
            let datExists = FileManager.default.fileExists(atPath: datURL.path)

            // Jeśli JSON istnieje, ale brakuje HEA/DAT
            if jsonExists && (!heaExists || !datExists) {
                if let ecg = ECGLoader.loadMultiLead(from: jsonURL, format: .json),
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
                        print("❌ Failed to create .hea/.dat from JSON for \(base): \(error)")
                    }
                } else {
                    print("❌ Could not load JSON or missing signal for \(base)")
                }
            }

            // Jeśli HEA i DAT istnieją, ale brak JSON
            if heaExists && datExists && !jsonExists {
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
                        print("❌ Failed to create .json from HEA for \(base): \(error)")
                    }
                } else {
                    print("❌ Could not load HEA or missing signal for \(base)")
                }
            }
        }
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
            try? FileManager.default.removeItem(at: url)
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








