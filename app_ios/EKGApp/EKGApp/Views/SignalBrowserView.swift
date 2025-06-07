import SwiftUI
import Charts
import UniformTypeIdentifiers


struct SignalBrowserView: View {
    @State private var recordings: [ECGRecordingSet] = []
    @State private var showShareSheet = false
    @State private var fileToShareURLs: [URL] = []
    @State private var selectedFileToOpen: String? = nil
    @State private var showAlert = false
    
    
    

    var body: some View {
        NavigationView {
            List {
                ForEach(recordings) { rec in
                    HStack(spacing: 16) {
                        
                        
                        Menu {
                            Button("📂 Open JSON") {
                                checkFileAndNavigate(rec.json, format: .json)
                            }
                            Button("📂 Open WFDB") {
                                checkFileAndNavigate(rec.wfdbHea, format: .wfdb)
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
                    Button(action: loadRecordings) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }

            // Navigation
            .background(
                NavigationLink(
                    destination: selectedFileToOpen.map { SignalPlaybackView(filename: $0) },
                    isActive: Binding(
                        get: { selectedFileToOpen != nil },
                        set: { if !$0 { selectedFileToOpen = nil } }
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
        if let _ = ECGLoader.loadMultiLead(from: fileURL, format: format) {
            selectedFileToOpen = fileURL.lastPathComponent
        } else {
            showAlert = true
        }
    }



    private func loadRecordings() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let allFiles = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey], options: .skipsHiddenFiles)

            let jsons = allFiles.filter { $0.pathExtension == "json" }
            var sets: [ECGRecordingSet] = []

            for json in jsons {
                let base = json.deletingPathExtension().lastPathComponent
                let dat = dir.appendingPathComponent("\(base).dat")
                let hea = dir.appendingPathComponent("\(base).hea")
                sets.append(.init(baseName: base, json: json, wfdbDat: dat, wfdbHea: hea))
            }

            recordings = sets.sorted {
                let d1 = (try? $0.json.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let d2 = (try? $1.json.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
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








