import SwiftUI
import Charts
import Combine

struct SignalRecorderView: View {
    @StateObject private var ble = EKGBLEManager()
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
    let samplesPerTick = 10
    let updateInterval = 0.05
    let maxSamples = 5000

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }

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
                            .frame(height: geometry.size.height * 0.35)
                            .background(backgroundColor)
                            .cornerRadius(10)
                            .padding(.horizontal)

                            HStack(spacing: 20) {
                                Button(isProcessing ? "Stop" : "Start") {
                                    isProcessing ? stopStream() : startStream()
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(isProcessing ? .red : .green)
                                .disabled(!ble.isDeviceValid && !isProcessing) // 🚫 Zablokuj, jeśli nieprzetestowane


                                Button(isRecording ? "Stop Recording" : "Record") {
                                    if isRecording {
                                        recordingEndTime = Date()
                                        saveRecording()
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

    private func startStream() {
        samples = []
        recordingBuffer = []
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
        guard isProcessing, !ble.rawBuffer.isEmpty else { return }

        var chunk: [Float] = []
        for _ in 0..<samplesPerTick {
            if let v = ble.rawBuffer.first {
                ble.rawBuffer.removeFirst()
                samples.append(v)
                chunk.append(v)
            }
        }

        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }

        if isRecording {
            recordingBuffer.append(contentsOf: chunk)
        }
    }

    
    
    
    private func saveRecording() {
        let fs = settings.sampleRateIn
        let start = recordingStartTime ?? Date()
        let end = recordingEndTime ?? Date()

        // ❌ Blokada jeśli < 1 sekunda
        guard recordingBuffer.count >= fs else {
            saveErrorMessage = "Recording too short. Must be at least 1 second."
            showSaveError = true
            return
        }

        let formatter = ISO8601DateFormatter()
        let startStr = formatter.string(from: start)
        let endStr = formatter.string(from: end)

        let jsonObject: [String: Any] = [
            "fs": fs,
            "lead": "II",
            "start_time": startStr,
            "end_time": endStr,
            "signal": recordingBuffer
        ]

        do {
            let data = try JSONSerialization.data(withJSONObject: jsonObject, options: .prettyPrinted)
            let fileFormatter = DateFormatter()
            fileFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            let filename = "ecg_\(fileFormatter.string(from: start)).json"
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)

            try data.write(to: url)
            print("✅ Saved to \(url)")
            showSaveSuccess = true
        } catch {
            saveErrorMessage = "Error saving file: \(error.localizedDescription)"
            showSaveError = true
        }
    }

    
    
}

