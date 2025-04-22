import SwiftUI
import CoreBluetooth
import Accelerate
import CoreML
import UniformTypeIdentifiers
import Charts

// MARK: - ContentView

struct ChartDataPoint: Identifiable {
    let id = UUID()
    let x: Int
    let y: Float
}

struct ContentView: View {
    @StateObject private var ble = EKGBLEManager()
    @StateObject private var rec = EKGRecorder()
    private let classifier = EKGClassifier()
    @State private var mode = 0
    @State private var importedData = [Float]()
    @State private var result = ""
    @State private var isRecording = false
    @State private var statusMessage = ""

    // Buffers for R-peak detection
    @State private var buffer: [Float] = []
    let fs=360
    let peakThreshold: Float = 0.5
    let minInterval =  Int(0.2 * 360) // 200ms

    var body: some View {
        VStack(spacing: 16) {
            Picker("Mode", selection: $mode) {
                Text("Live BLE").tag(0)
                Text("Import .dat").tag(1)
                Text("Playback").tag(2)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding([.horizontal, .top])

            Group {
                switch mode {
                case 0: liveBLEView
                case 1: importView
                default: playbackView
                }
            }
            .padding(.horizontal)

            Text(statusMessage)
                .font(.footnote)
                .foregroundColor(.gray)
                .padding(.bottom)
        }
        .onAppear {
            NotificationCenter.default.addObserver(
                forName: .didImportData,
                object: nil,
                queue: .main
            ) { notification in
                importedData = notification.object as? [Float] ?? []
                statusMessage = "Imported \(importedData.count) samples"
            }
        }
    }

    // MARK: - Live BLE View
    var liveBLEView: some View {
        VStack {
            Chart(rec.recorded.enumerated().map { ChartDataPoint(x: $0.offset, y: $0.element) }) { point in
                LineMark(
                    x: .value("Index", point.x),
                    y: .value("Voltage", point.y)
                )
                PointMark(
                    x: .value("Index", point.x),
                    y: .value("Voltage", point.y)
                )
            }
            .chartYScale(domain: -1...1)
            .frame(height: 200)

            HStack {
                Button(isRecording ? "Stop" : "Start") {
                    isRecording ? stopProcessing() : startProcessing()
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 16)
                .background(isRecording ? Color.red.opacity(0.7) : Color.green.opacity(0.7))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
        }
        .onReceive(ble.$rawBuffer) { buf in
            guard mode==0, isRecording else { return }
            for v in buf {
                rec.append(v)
                processSample(v)
            }
        }
    }

    // MARK: - Import View
    var importView: some View {
        Button("Import .dat") { importDAT() }
            .padding()
    }

    // MARK: - Playback View
    var playbackView: some View {
        VStack(spacing: 12) {
            Button("Classify Imported") { classify(importedData) }
                .padding()
            Text("Result: \(result)")
        }
    }

    // MARK: - Actions
    func importDAT() {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.data])
        picker.allowsMultipleSelection = false
        picker.delegate = Context.shared
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController {
            root.present(picker, animated: true)
        }
    }

    func classify(_ data: [Float]) {
        let rs = Resampler.resample(input: data, srcRate: 128, dstRate: Float(fs))
        result = classifier.predict(rs)
        statusMessage = "Classification done"
    }

    func startProcessing() {
        rec.start()
        buffer.removeAll()
        isRecording = true
        statusMessage = "Running..."
    }

    func stopProcessing() {
        isRecording = false
        statusMessage = "Stopped"
    }

    // MARK: - R-peak detection & segment classify
    func processSample(_ sample: Float) {
        buffer.append(sample)
        if buffer.count < fs { return }
        // simple threshold detect
        let n=buffer.count
        if buffer[n-1] > peakThreshold,
           n>minInterval,
           buffer[n-1] > buffer[n-2] {
            // found R peak at last sample
            let start = max(0, n - Int(fs/2))
            let end = min(n, start + fs)
            let segment = Array(buffer[start..<end])
            let rs = Resampler.resample(input: segment, srcRate: Float(fs), dstRate: Float(fs))
            let label = classifier.predict(rs)
            statusMessage = "Peak @\(n), class=\(label)"
            // shift buffer to avoid re-detect
            buffer = Array(buffer[(n-minInterval)...])
        }
        // keep buffer size <= 5s
        if buffer.count>fs*5 { buffer.removeFirst(buffer.count-fs*5) }
    }
}

