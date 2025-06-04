import SwiftUI
import Charts
import UniformTypeIdentifiers

struct SignalBrowserView: View {
    @State private var filenames: [String] = []
    @State private var showShareSheet = false
    @State private var fileToShareURL: URL? = nil

    @State private var selectedFileToOpen: String? = nil
    @State private var showAlert = false

    var body: some View {
        NavigationView {
            VStack {
                List {
                    ForEach(filenames, id: \.self) { filename in
                        HStack(spacing: 16) {
                            Button {
                                checkFileAndNavigate(filename)
                            } label: {
                                Text(filename)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                            }

                            Spacer()

                            Button {
                                shareFile(named: filename)
                            } label: {
                                Image(systemName: "square.and.arrow.up")
                                    .resizable()
                                    .frame(width: 24, height: 24)
                                    .foregroundColor(.blue)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(BorderlessButtonStyle())

                            Button {
                                deleteFile(named: filename)
                            } label: {
                                Image(systemName: "trash")
                                    .resizable()
                                    .frame(width: 24, height: 24)
                                    .foregroundColor(.red)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(BorderlessButtonStyle())
                        }
                        .padding(.vertical, 6)
                    }
                    .onDelete(perform: deleteFiles)
                }

                // Ukryty NavigationLink aktywowany tylko gdy plik OK
                NavigationLink(
                    destination: selectedFileToOpen.map { SignalPlaybackView(filename: $0) },
                    isActive: Binding(
                        get: { selectedFileToOpen != nil },
                        set: { if !$0 { selectedFileToOpen = nil } }
                    )
                ) {
                    EmptyView()
                }
                .hidden()

            }

            .navigationTitle("Saved Records")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    EditButton()
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: loadFilenames) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh file list")
                }
            }

            .onAppear(perform: loadFilenames)
            .sheet(isPresented: $showShareSheet) {
                if let url = fileToShareURL {
                    ActivityView(activityItems: [url])
                }
            }

            .alert("Cant open fle", isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("This ECG recording is not valid and cannot be viewed.")
            }
        }
    }

    private func checkFileAndNavigate(_ filename: String) {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = dir.appendingPathComponent(filename)

        do {
            let data = try Data(contentsOf: fileURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let fs = json["fs"] as? Int,
               let signal = json["signal"] as? [Double] {
                if signal.count >= fs {
                    selectedFileToOpen = filename
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

    private func loadFilenames() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let urls = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey], options: .skipsHiddenFiles)
            let jsonFiles = urls.filter { $0.pathExtension == "json" }

            let sortedFiles = jsonFiles.sorted {
                let date1 = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let date2 = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return date1 > date2
            }

            filenames = sortedFiles.map { $0.lastPathComponent }
        } catch {
            print("❌ Error reading files: \(error)")
        }
    }

    private func deleteFile(named filename: String) {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = dir.appendingPathComponent(filename)
        do {
            try FileManager.default.removeItem(at: fileURL)
            print("🗑️ Deleted \(filename)")
            loadFilenames()
        } catch {
            print("❌ Error deleting \(filename): \(error)")
        }
    }

    private func deleteFiles(at offsets: IndexSet) {
        for index in offsets {
            let filename = filenames[index]
            deleteFile(named: filename)
        }
    }

    private func shareFile(named filename: String) {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileToShareURL = dir.appendingPathComponent(filename)
        if fileToShareURL != nil {
            showShareSheet = true
        } else {
            print("❌ Share file URL nil")
        }
    }
}

// UIKit wrapper do natywnego arkusza udostępniania
struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
        controller.completionWithItemsHandler = { activityType, completed, returnedItems, error in
            if completed {
                print("✅ Shared successfully")
            } else {
                print("ℹ️ Share cancelled or failed")
            }
            if let error = error {
                print("❌ Share error: \(error)")
            }
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

