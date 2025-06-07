import SwiftUI
import Charts
import UniformTypeIdentifiers

struct ECGRecordingSet: Identifiable {
    var id: String { baseName }
    let baseName: String
    let json: URL
    let wfdbDat: URL
    let wfdbHea: URL
    let mat: URL
}

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
                        Button {
                            checkFileAndNavigate(rec.json)
                        } label: {
                            Text(rec.baseName + ".json")
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
                            Button("📤 MATLAB (.mat)") { shareFiles(urls: [rec.mat]) }
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundColor(.blue)
                        }
                        .buttonStyle(BorderlessButtonStyle())

                        Button {
                            deleteRecording(rec)
                        } label: {
                            Image(systemName: "trash")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundColor(.red)
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

    private func checkFileAndNavigate(_ fileURL: URL) {
        do {
            let data = try Data(contentsOf: fileURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let fs = json["fs"] as? Int,
               let signal = json["signal"] as? [Double] {
                if signal.count >= fs {
                    selectedFileToOpen = fileURL.lastPathComponent
                } else {
                    showAlert = true
                }
            } else {
                showAlert = true
            }
        } catch {
            print("❌ File load error: \(error)")
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
                let mat = dir.appendingPathComponent("\(base).mat")
                sets.append(.init(baseName: base, json: json, wfdbDat: dat, wfdbHea: hea, mat: mat))
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
        for url in [rec.json, rec.wfdbDat, rec.wfdbHea, rec.mat] {
            try? FileManager.default.removeItem(at: url)
        }
        loadRecordings()
    }

    private func deleteAtOffsets(_ offsets: IndexSet) {
        for index in offsets {
            deleteRecording(recordings[index])
        }
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

