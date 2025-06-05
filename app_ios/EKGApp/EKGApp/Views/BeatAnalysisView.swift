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
    
    @State private var showInvalidDeviceAlert = false
    
    @Environment(\.dismiss) var dismiss
    @ObservedObject private var settings = AppSettings.shared
    
    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }
    var chartColor: Color { settings.darkModeEnabled ? .cyan : .blue }
    
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
                                .disabled(!ble.isDeviceValid && !isProcessing)
                                .opacity((!ble.isDeviceValid && !isProcessing) ? 0.4 : 1.0)

                                
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
        
        
        .onChange(of: ble.connectedPeripheral) {
            if ble.connectedPeripheral != nil {
                ble.startScanWithTimeoutAfterConnect {
                    showInvalidDeviceAlert = true
                }
            }
        }
        
        
        
        .onAppear {
            ble.startScan()
        }
        
        .onDisappear {
            stopProcessing()
            ble.reset()
        }
        
        .alert("Invalid Device", isPresented: $showInvalidDeviceAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("❌ Please choose a valid ECG monitor device")
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
        let fs: Float = 360  // ✅ bo R-peaki są wykrywane po resamplingu
        
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
        // 1. Resample from 128 Hz → 360 Hz
        let resampled = Resampler.resample(input: window, srcRate: 128, dstRate: 360)

        // 2. Detect R-peaks
        let peaks = RPeakDetector.detectRPeaks(signal: resampled, fs: 360, sensitivity: 0.7)
        var classifiedPeaks: [Int] = []

        for localPeak in peaks {
            // 3. Wytnij segment 540 próbek (±270)
            var segment: [Float] = []
            for i in (localPeak - 270)...(localPeak + 269) {
                if i >= 0 && i < resampled.count {
                    segment.append(resampled[i])
                } else {
                    segment.append(0.0)
                }
            }

            // 4. Upewnij się, że segment ma dokładnie 540 próbek
            if segment.count != 540 {
                segment = Array(segment.prefix(540))
                if segment.count < 540 {
                    segment += Array(repeating: 0.0, count: 540 - segment.count)
                }
            }

            // 5. Przekaż do klasyfikatora
            let predicted = classifier.predict(input: segment)
            if predicted != "?" {
                classCounts[predicted, default: 0] += 1
                qrsCount += 1
                classifiedPeaks.append(localPeak)
            }
        }
        print("🔍 \(peaks.count) peaks in resampled window (\(resampled.count) samples)")
        print("🔍 \(classifiedPeaks.count) classified peaks")


        // 6. Aktualizuj BPM
        updateBPM(from: classifiedPeaks)
    }

    
}
