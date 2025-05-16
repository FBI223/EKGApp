


import SwiftUI
import Charts
import Combine


struct RhythmAnalysisView: View {
    @StateObject private var ble = EKGBLEManager()

    @State private var rhythmBuffer: [Float] = []
    @State private var prediction: String = "—"
    @State private var rhythmClassCounts: [String: Int] = ["NSR": 0, "AFib": 0, "AFL": 0, "VT": 0, "Other": 0]
    @State private var timer: AnyCancellable?
    @State private var isProcessing = false

    let windowLength = 1280 // 10s * 128Hz
    private let model = EKGClassifier()

    var body: some View {
        VStack(spacing: 10) {
            if ble.connectedPeripheral == nil {
                List(ble.devices, id: \.identifier) { device in
                    Button { ble.connect(to: device) }
                    label: { Text(device.name ?? "Unknown") }
                }
            } else {
                Chart {
                    let visible = Array(rhythmBuffer.suffix(windowLength))
                    ForEach(0..<windowLength, id: \.self) { i in
                        let value = i < visible.count ? visible[i] : 0.0
                        LineMark(x: .value("Index", i), y: .value("Voltage", value))
                    }
                }
                .chartXScale(domain: 0...windowLength)
                .chartYScale(domain: -3...3)
                .frame(height: 250)
                .padding()

                Text("Rhythm Prediction: \(prediction)").font(.headline)
                HStack {
                    ForEach(["NSR", "AFib", "AFL", "VT", "Other"], id: \.self) { k in
                        Text("\(k): \(rhythmClassCounts[k] ?? 0)").font(.subheadline)
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
                        rhythmBuffer = []
                        prediction = "—"
                        rhythmClassCounts = ["NSR": 0, "AFib": 0, "AFL": 0, "VT": 0, "Other": 0]
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.gray)
                }
            }
        }
        .padding()
        .onAppear {
            ble.disconnect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                ble.startScan()
            }
        }
        .onDisappear { stopProcessing() }
    }

    func startProcessing() {
        rhythmBuffer = []
        prediction = "—"
        rhythmClassCounts = ["NSR": 0, "AFib": 0, "AFL": 0, "VT": 0, "Other": 0]
        isProcessing = true

        timer = Timer.publish(every: 0.05, on: .main, in: .common)
            .autoconnect()
            .sink { _ in updateRhythm() }
    }

    func stopProcessing() {
        isProcessing = false
        timer?.cancel()
    }

    func updateRhythm() {
        guard isProcessing, !ble.rawBuffer.isEmpty else { return }

        for _ in 0..<10 {
            if let v = ble.rawBuffer.first {
                ble.rawBuffer.removeFirst()
                rhythmBuffer.append(v)
                if rhythmBuffer.count > 5000 {
                    rhythmBuffer.removeFirst(rhythmBuffer.count - 5000)
                }
            }
        }

        if rhythmBuffer.count >= windowLength {
            let segment = Array(rhythmBuffer.suffix(windowLength))
            let possibleClasses = ["NSR", "AFib", "AFL", "VT", "Other"]
            if let randomClass = possibleClasses.randomElement() {
                prediction = randomClass
                rhythmClassCounts[randomClass, default: 0] += 1
                print("[Rhythm] 10s randomly classified as \(randomClass)")
            }
            rhythmBuffer.removeFirst(windowLength)
        }
    }
}
