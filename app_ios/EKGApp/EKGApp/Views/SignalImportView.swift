import SwiftUI
import UniformTypeIdentifiers

struct SignalImportView: UIViewControllerRepresentable {
    var onImport: ([URL]) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onImport: onImport)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // Dozwolone typy: tylko JSON, HEA, DAT
        let json = UTType(filenameExtension: "json") ?? .json
        let hea = UTType(filenameExtension: "hea") ?? .plainText  // tekstowy nagłówek
        let dat = UTType(filenameExtension: "dat") ?? .data       // binarny plik sygnału

        let picker = UIDocumentPickerViewController(
            forOpeningContentTypes: [json, hea, dat],
            asCopy: true
        )

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
            let allowedExtensions = ["json", "hea", "dat"]

            // Filtrowanie tylko dopuszczalnych plików
            let filtered = urls.filter { url in
                allowedExtensions.contains(url.pathExtension.lowercased())
            }

            onImport(filtered)
        }
    }
}

