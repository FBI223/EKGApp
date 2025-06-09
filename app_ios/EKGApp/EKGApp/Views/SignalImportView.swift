import SwiftUI
import UniformTypeIdentifiers

struct SignalImportView: UIViewControllerRepresentable {
    var onImport: ([URL]) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onImport: onImport)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let hea = UTType(filenameExtension: "hea") ?? .data
        let dat = UTType(filenameExtension: "dat") ?? .data
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.json, hea, dat], asCopy: true)
        picker.allowsMultipleSelection = true
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        var onImport: ([URL]) -> Void
        init(onImport: @escaping ([URL]) -> Void) {
            self.onImport = onImport
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            onImport(urls)
        }
    }
}

