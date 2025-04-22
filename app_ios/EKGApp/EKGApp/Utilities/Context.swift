import UIKit
import UniformTypeIdentifiers

class Context: NSObject, UIDocumentPickerDelegate {
    static let shared = Context()
    func documentPicker(_ picker: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let u = urls.first,
              let data = try? Data(contentsOf: u) else { return }
        let count = data.count / MemoryLayout<Float>.size
        var arr = [Float](repeating: 0, count: count)
        _ = arr.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        // zapisz do singletona lub użyj NotificationCenter,
        // zamiast tworzyć nowy ContentView()
        NotificationCenter.default.post(
          name: .didImportData, object: arr
        )
    }
}

extension Notification.Name {
    static let didImportData = Notification.Name("didImportData")
}
