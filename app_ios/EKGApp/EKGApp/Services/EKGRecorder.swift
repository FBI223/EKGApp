import SwiftUI
import CoreBluetooth
import Accelerate
import CoreML
import UniformTypeIdentifiers




// MARK: - Recorder

class EKGRecorder: ObservableObject {
    @Published var recorded = [Float]()
    func start() { recorded.removeAll() }
    func append(_ v: Float) { recorded.append(v) }
    func save(to url: URL) throws {
        let data = recorded.reduce(into: Data()) {
            var x = $1
            $0.append(UnsafeBufferPointer(start: &x, count: 1))
        }
        try data.write(to: url)
    }
}
