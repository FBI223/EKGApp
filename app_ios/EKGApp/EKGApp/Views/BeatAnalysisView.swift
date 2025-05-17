import SwiftUI
import Charts
import Combine

struct BeatAnalysisView: View {
    @StateObject private var ble = EKGBLEManager()
    @State private var samples: [Float] = []
    @State private var recordingBuffer: [Float] = []
    @State private var timer: AnyCancellable?
    @State private var isProcessing = false
    
    @State private var lastDetectedTimes: [Date] = []
    @State private var bpm: Int = 0


    @Environment(\.dismiss) var dismiss
    @ObservedObject private var settings = AppSettings.shared

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }
    var chartColor: Color { settings.darkModeEnabled ? .cyan : .blue }

    @State private var fs_main = 128
    @State private var qrsCount = 0
    @State private var classCounts: [String: Int] = ["N": 0, "S": 0, "V": 0, "F": 0, "Q": 0]

    let maxWindowSize = 700
    let maxTotalSamples = 5000
    let samplesPerTick = 10
    let updateInterval = 0.05
    let segmentRadius = 96

    private let classifier = EKGClassifier()

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 16) {
                    if ble.connectedPeripheral == nil {
                        VStack(spacing: 10) {
                            Text("Select ECG Device")
                                .font(.headline)
                                .foregroundColor(foregroundColor)

                            List(ble.devices, id: \.identifier) { device in
                                Button {
                                    ble.connect(to: device)
                                } label: {
                                    HStack {
                                        Image(systemName: "antenna.radiowaves.left.and.right")
                                        Text(device.name ?? "Unknown")
                                    }
                                    .padding(8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .foregroundColor(foregroundColor)
                            }
                            .listStyle(.plain)
                            .frame(height: geometry.size.height * 0.5)
                            .background(backgroundColor)
                        }
                        .padding(.horizontal)
                    } else {
                        VStack(spacing: 12) {
                            Chart {
                                let visible = Array(samples.suffix(maxWindowSize))
                                ForEach(0..<visible.count, id: \.self) { i in
                                    LineMark(
                                        x: .value("Index", i),
                                        y: .value("Voltage", visible[i])
                                    )
                                    .foregroundStyle(chartColor)
                                }
                            }
                            .chartXScale(domain: 0...maxWindowSize)
                            .chartYScale(domain: Double(settings.yAxisRange.lowerBound)...Double(settings.yAxisRange.upperBound))
                            .frame(height: geometry.size.height * 0.35)
                            .background(backgroundColor)
                            .cornerRadius(10)
                            .padding(.horizontal)

                            Text("❤️ BPM: \(bpm)")
                                .font(.title3)
                                .bold()
                                .foregroundColor(.red)

                            Text("QRS Beat Count: \(qrsCount)")
                                .font(.title3)
                                .foregroundColor(foregroundColor)

                            HStack {
                                ForEach(["N", "S", "V", "F", "Q"], id: \.self) { k in
                                    VStack {
                                        Text(k)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text("\(classCounts[k] ?? 0)")
                                            .font(.headline)
                                            .foregroundColor(foregroundColor)
                                    }
                                    .frame(maxWidth: .infinity)
                                }
                            }

                            HStack(spacing: 20) {
                                Button(isProcessing ? "Stop" : "Start") {
                                    isProcessing ? stopProcessing() : startProcessing()
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(isProcessing ? .red : .green)

                                Button("Disconnect") {
                                    stopProcessing()
                                    ble.reset()
                                    samples = []
                                    qrsCount = 0
                                    classCounts = ["N": 0, "S": 0, "V": 0, "F": 0, "Q": 0]
                                    dismiss()
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.gray)
                            }
                        }
                        .padding(.horizontal)
                    }
                }
                .frame(minHeight: geometry.size.height)
            }
            .background(backgroundColor.ignoresSafeArea())
        }
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
        .onAppear {
            ble.startScan()
        }
        
        .onDisappear {
            stopProcessing()
            ble.reset()
        }
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
    
    private func updateBPM(from qrsPeaks: [Int]) {
        let fs = Float(settings.sampleRateIn)

        guard qrsPeaks.count >= 2 else {
            bpm = 0
            return
        }

        let rrIntervals: [Float] = zip(qrsPeaks.dropFirst(), qrsPeaks).map { Float($0 - $1) / fs }
        let avgRR = rrIntervals.reduce(0, +) / Float(rrIntervals.count)
        
        bpm = avgRR > 0 ? Int(60.0 / avgRR) : 0
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

        while samples.count >= maxWindowSize {
            let window = Array(samples.prefix(maxWindowSize))
            classifyWindow(window)
            samples.removeFirst(maxWindowSize)
        }
    }

    private func classifyWindow(_ window: [Float]) {
        let peaks = RPeakDetector.detectRPeaks(signal: window, fs: fs_main)
        for localPeak in peaks {
            let left = localPeak - segmentRadius
            let right = localPeak + segmentRadius

            var segment: [Float] = []
            for i in left...right {
                segment.append(i >= 0 && i < window.count ? window[i] : 0.0)
            }

            guard segment.count == 2 * segmentRadius + 1 else { continue }

            qrsCount += 1
            let predicted = classifier.predict(input: segment)
            print("QRS at local index \(localPeak): \(predicted)")
            classCounts[predicted, default: 0] += 1
            
            
            
            
            updateBPM(from: peaks)

        }
    }
}

