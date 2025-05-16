

import SwiftUI
import Charts
import Combine

struct BeatAnalysisView: View {
    @StateObject private var ble = EKGBLEManager()
    
    @State private var samples: [Float] = []                // bufor próbek
    @State private var recordingBuffer: [Float] = []        // do zapisu
    @State private var timer: AnyCancellable?
    @State private var isProcessing = false

    @State private var fs_main = 128
    @State private var qrsCount = 0
    @State private var classCounts: [String: Int] = ["N": 0, "S": 0, "V": 0, "F": 0, "Q": 0]

    let maxWindowSize = 700        // rozmiar okna do klasyfikacji
    let maxTotalSamples = 5000     // limit całego bufora
    let samplesPerTick = 10
    let updateInterval = 0.05      // 50 ms
    let segmentRadius = 96         // ±96 próbek = 192

    private let classifier = EKGClassifier()

    var body: some View {
        VStack(spacing: 10) {
            if ble.connectedPeripheral == nil {
                List(ble.devices, id: \.identifier) { device in
                    Button { ble.connect(to: device) }
                    label: { Text(device.name ?? "Unknown") }
                }
            } else {
                Chart {
                    // pokaż ostatnie maxWindowSize próbek
                    let visible = Array(samples.suffix(maxWindowSize))
                    ForEach(0..<maxWindowSize, id: \.self) { i in
                        let value = i < visible.count ? visible[i] : 0.0
                        LineMark(x: .value("Index", i), y: .value("Voltage", value))
                    }

                }
                .chartXScale(domain: 0...maxWindowSize) // wymusza zakres X
                .chartYScale(domain: -3...3)
                .frame(height: 250)
                .padding()

                Text("QRS Beat Count: \(qrsCount)").font(.headline)
                HStack {
                    ForEach(["N","S","V","F","Q"], id: \.self) { k in
                        Text("\(k): \(classCounts[k] ?? 0)").font(.subheadline)
                    }
                }

                HStack {
                    Button(isProcessing ? "Stop" : "Start") {
                        isProcessing ? stopProcessing() : startProcessing()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isProcessing ? .red : .green)

                    Button("Disconnect") {
                        ble.disconnect()
                        stopProcessing()
                        samples = []
                        qrsCount = 0
                        classCounts = ["N": 0, "S": 0, "V": 0, "F": 0, "Q": 0]
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.gray)
                }
            }
        }
        .onAppear { ble.startScan() }
    }

    func startProcessing() {
        samples = []
        recordingBuffer = []
        qrsCount = 0
        classCounts = ["N": 0, "S": 0, "V": 0, "F": 0, "Q": 0]
        isProcessing = true

        timer = Timer.publish(every: updateInterval, on: .main, in: .common)
            .autoconnect()
            .sink { _ in updateSamples() }
    }

    func stopProcessing() {
        isProcessing = false
        timer?.cancel()
    }

    private func updateSamples() {
        guard isProcessing, !ble.rawBuffer.isEmpty else { return }

        for _ in 0..<samplesPerTick {
            if let v = ble.rawBuffer.first {
                ble.rawBuffer.removeFirst()
                samples.append(v)
                if samples.count > maxTotalSamples {
                    samples.removeFirst(samples.count - maxTotalSamples)
                }
            }
        }

        // gdy uzbieramy pełne okno, klasyfikujemy wszystkie QRS w danym oknie
        while samples.count >= maxWindowSize {
            let window = Array(samples.prefix(maxWindowSize))
            classifyWindow(window)
            // usuń cały ten fragment, przesuwając bufor
            samples.removeFirst(maxWindowSize)
        }
    }


    
    private func classifyWindow(_ window: [Float]) {
        let peaks = RPeakDetector.detectRPeaks(signal: window, fs: fs_main)

        for localPeak in peaks {
            let left = localPeak - segmentRadius
            let right = localPeak + segmentRadius

            // pozwalamy segmentowi wyjść poza okno, ale dopełniamy zerami
            var segment: [Float] = []
            for i in left...right {
                if i >= 0 && i < window.count {
                    segment.append(window[i])
                } else {
                    segment.append(0.0) // dopełnienie
                }
            }

            guard segment.count == 2 * segmentRadius + 1 else { continue }

            qrsCount += 1
            let predicted = classifier.predict(input128Hz: segment)
            print("QRS at local index \(localPeak): \(predicted)")
            classCounts[predicted, default: 0] += 1
        }
    }
    
    
    
}
