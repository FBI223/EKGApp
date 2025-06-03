// Wersja z ikonami 24x24, by było więcej miejsca na nazwę pliku

import SwiftUI
import Charts
import UniformTypeIdentifiers

struct SignalBrowserView: View {
    @State private var filenames: [String] = []
    @State private var showShareSheet = false
    @State private var fileToShareURL: URL? = nil

    var body: some View {
        NavigationView {
            List {
                ForEach(filenames, id: \.self) { filename in
                    HStack(spacing: 16) {
                        NavigationLink(destination: SignalPlaybackView(filename: filename)) {
                            Text(filename)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .font(.headline)
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
            .navigationTitle("Saved Records")
            .toolbar {
                EditButton()
            }
            .onAppear(perform: loadFilenames)
            .sheet(isPresented: $showShareSheet) {
                if let url = fileToShareURL {
                    ActivityView(activityItems: [url])
                }
            }
        }
    }

    private func loadFilenames() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            filenames = files.filter { $0.hasSuffix(".json") && $0.hasPrefix("signal_") }
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

