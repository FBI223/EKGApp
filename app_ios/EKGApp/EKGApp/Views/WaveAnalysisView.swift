import SwiftUI
import Charts
import CoreML
import Combine

struct WaveAnalysisView: View {
    @StateObject private var ble = EKGBLEManager()
    @ObservedObject private var settings = AppSettings.shared

    @State private var samples: [Float] = []
    @State private var predictedClasses: [Int] = []
    @State private var isProcessing = false
    @State private var timer: AnyCancellable?

    private let classifier = WaveformClassifier()
    private let targetLength = 2000  // 4s @ 500 Hz
    private let classLabels = ["None", "P", "QRS", "T"]
    private let classColors: [Color] = [.gray, .green, .red, .blue]

    @State private var showInvalidDeviceAlert = false
    @Environment(\.dismiss) var dismiss

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 16) {
                    if ble.connectedPeripheral == nil {
                        VStack(spacing: 10) {
                            Text("Select ECG Device")
                                .font(.headline)
                                .foregroundColor(.primary)

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
                            }
                            .listStyle(.plain)
                            .frame(height: geometry.size.height * 0.5)
                        }
                        .padding(.horizontal)
                    } else {
                        VStack(spacing: 12) {
                            Text("🔬 Waveform Classification")
                                .font(.title3)

                            Chart {
                                let visible = Array(samples)
                                ForEach(Array(visible.enumerated()), id: \.offset) { i, v in
                                    let label = i < predictedClasses.count ? predictedClasses[i] : 0
                                    PointMark(
                                        x: .value("Time", i),
                                        y: .value("Signal", v)
                                    )
                                    .foregroundStyle(classColors[label])
                                }
                            }
                            .frame(height: 250)
                            .padding()
                            
                            // === Legenda ===
                             HStack(spacing: 12) {
                                 ForEach(0..<classLabels.count, id: \.self) { label in
                                     Label {
                                         Text(classLabels[label])
                                     } icon: {
                                         Circle()
                                             .fill(classColors[label])
                                             .frame(width: 12, height: 12)
                                     }
                                     .labelStyle(.titleAndIcon)
                                 }
                             }
                             .font(.caption)
                             .padding(.bottom, 8)
                            
                            
                            HStack(spacing: 16) {
                                Button(isProcessing ? "Stop" : "Start") {
                                    isProcessing ? stop() : start()
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(isProcessing ? .red : .green)

                                Button("Disconnect") {
                                    stop()
                                    ble.reset()
                                    samples = []
                                    predictedClasses = []
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
            stop()
            ble.reset()
        }
        .alert("Invalid Device", isPresented: $showInvalidDeviceAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("❌ Please choose a valid ECG monitor device")
        }
    }

    func start() {
        samples = []
        predictedClasses = []
        isProcessing = true

        timer = Timer.publish(every: 0.05, on: .main, in: .common)
            .autoconnect()
            .sink { _ in processSamples() }
    }

    func stop() {
        isProcessing = false
        timer?.cancel()
    }

    func processSamples() {
        let inputFs = settings.sampleRateIn
        let inputLen = 512  // zamiast 4s – reaguj od razu

        guard ble.rawBuffer.count >= inputLen else { return }

        let segment = Array(ble.rawBuffer.prefix(inputLen))
        ble.rawBuffer.removeFirst(inputLen)

        var resampled = Resampler.resample(input: segment, srcRate: Float(inputFs), dstRate: 500.0)

        if resampled.count > targetLength {
            resampled = Array(resampled.prefix(targetLength))
        } else if resampled.count < targetLength {
            let padding = Array(repeating: Float(0.0), count: targetLength - resampled.count)
            resampled += padding
        }

        samples = resampled
        predictedClasses = classifier.predict(input: resampled)

        print("🔁 samples.count = \(samples.count)")
        print("✅ predictedClasses.count = \(predictedClasses.count)")
        print("📊 First 10 class predictions: \(predictedClasses.prefix(10))")
    }
}

