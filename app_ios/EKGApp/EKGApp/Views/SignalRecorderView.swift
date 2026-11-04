import SwiftUI
import Charts
import Combine

private let signalGain: Float = 200.0  // µV/LSB, typowo 200
private let leadName = "II"
private let resolutionBits = 16


struct SignalRecorderView: View {
    @StateObject private var ble = EKGBLEManager()

    /// Pass a signal to run this screen against the built-in device simulator instead of Bluetooth.
    init(simulatedSignal: SimulatedSignal? = nil) {
        _ble = StateObject(wrappedValue: EKGBLEManager(simulatedSignal: simulatedSignal))
    }
    @ObservedObject private var settings = AppSettings.shared

    @State private var samples: [Float] = []
    @State private var recordingBuffer: [Float] = []
    @State private var isProcessing = false
    @State private var isRecording = false
    @State private var timer: AnyCancellable?

    @State private var recordingStartTime: Date?
    @State private var recordingEndTime: Date?
    @State private var showInvalidDeviceAlert = false

    @State private var showSaveSuccess = false
    @State private var showSaveError = false
    @State private var saveErrorMessage = ""

    @Environment(\.dismiss) var dismiss

    let maxWindowSize = 1000
    let updateInterval = 0.05
    let maxSamples = 5000

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 16) {
                    if ble.isSimulation, ble.isConnected {
                        Label("Simulation: \(ble.connectedDeviceName ?? "")", systemImage: "testtube.2")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    if !ble.isConnected {
                        deviceSelectionView(height: geometry.size.height)
                    } else {
                        signalRecordingView(height: geometry.size.height)
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
            ble.disconnect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                ble.startScan()
            }
        }

        .onDisappear {
            stopStream()
            ble.reset()
        }

        .alert("Invalid Device", isPresented: $showInvalidDeviceAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("❌ Please choose a valid ECG monitor device")
        }

        .alert("✅ Saved", isPresented: $showSaveSuccess) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Recording saved successfully.")
        }

        .alert("❌ Save Error", isPresented: $showSaveError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(saveErrorMessage)
        }
    }

    // MARK: - Views

    private func deviceSelectionView(height: CGFloat) -> some View {
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
            .frame(height: height * 0.5)
            .background(backgroundColor)
        }
        .padding(.horizontal)
    }

    private func signalRecordingView(height: CGFloat) -> some View {
        VStack(spacing: 12) {
            Text("📡 Signal Recorder")
                .font(.title2)
                .bold()
                .foregroundColor(foregroundColor)

            Chart {
                let visible = Array(samples.suffix(maxWindowSize))
                ForEach(0..<visible.count, id: \.self) { i in
                    LineMark(x: .value("Index", i), y: .value("Value", visible[i]))
                }
            }
            .frame(height: height * 0.35)
            .background(backgroundColor)
            .cornerRadius(10)
            .padding(.horizontal)

            HStack(spacing: 20) {
                Button(isProcessing ? "Stop" : "Start") {
                    isProcessing ? stopStream() : startStream()
                }
                .buttonStyle(.borderedProminent)
                .tint(isProcessing ? .red : .green)
                .disabled(!ble.isDeviceValid && !isProcessing)

                Button(isRecording ? "Stop Recording" : "Record") {
                    if isRecording {
                        recordingEndTime = Date()
                        
                        do {
                            let fs = settings.sampleRateIn
                            let start = recordingStartTime ?? Date()
                            let end = recordingEndTime ?? Date()
                            let formatter = DateFormatter()
                            formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
                            let baseName = "ecg_record_\(formatter.string(from: start))"

                            try ECGFileSaver.saveAll(
                                baseName: baseName,
                                buffer: recordingBuffer,
                                fs: fs,
                                gain: signalGain,
                                leadName: leadName,
                                start: start,
                                end: end
                            )
                            
                            
                            showSaveSuccess = true

                            print("✅ Saved all formats")
                        } catch {
                            print("❌ Error saving recording: \(error)")
                        }

                        
                    } else {
                        recordingStartTime = Date()
                        recordingBuffer = []
                    }
                    isRecording.toggle()
                }
                .buttonStyle(.borderedProminent)
                .tint(isRecording ? .orange : .blue)
                .disabled(!isProcessing)

                Button("Disconnect") {
                    stopStream()
                    ble.reset()
                    samples = []
                    recordingBuffer = []
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .tint(.gray)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Logic

    private func startStream() {
        samples = []
        recordingBuffer = []
        ble.clearBuffer()
        isProcessing = true
        timer = Timer.publish(every: updateInterval, on: .main, in: .common)
            .autoconnect()
            .sink { _ in tick() }
    }

    private func stopStream() {
        isProcessing = false
        timer?.cancel()
    }

    private func tick() {
        guard isProcessing else { return }

        // Consume everything that arrived since the last tick so latency never accumulates
        let chunk = ble.drainBuffer()
        guard !chunk.isEmpty else { return }
        samples.append(contentsOf: chunk)

        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }

        if isRecording {
            recordingBuffer.append(contentsOf: chunk)
        }
    }




    // MARK: - Delete all files

    func deleteRecording(_ rec: ECGRecordingSet) {
        for url in [rec.json, rec.wfdbDat, rec.wfdbHea] {
            if FileManager.default.fileExists(atPath: url.path) {
                try? FileManager.default.removeItem(at: url)
                print("🗑️ Deleted \(url.lastPathComponent)")
            }
        }
    }
}

