import SwiftUI
import Charts
import Combine

struct SignalRecorderView: View {
    @StateObject private var ble = EKGBLEManager()
    @State private var samples: [Float] = []
    @State private var recordingBuffer: [Float] = []
    @State private var isProcessing = false
    @State private var isRecording = false
    @State private var timer: AnyCancellable?
    @ObservedObject private var settings = AppSettings.shared
    
    
    @State private var recordingStartTime: Date?
    @State private var recordingEndTime: Date?
    
    @State private var showInvalidDeviceAlert = false


    let maxWindowSize = 1000
    let samplesPerTick = 10
    let updateInterval = 0.05
    let maxSamples = 5000

    var body: some View {
        VStack(spacing: 20) {
            Text("📡 Signal Recorder")
                .font(.title2)
                .bold()

            Chart {
                let visible = Array(samples.suffix(maxWindowSize))
                ForEach(0..<visible.count, id: \.self) { i in
                    LineMark(x: .value("i", i), y: .value("v", visible[i]))
                }

            }
            .frame(height: 250)
            .background(Color.black.opacity(0.05))
            .cornerRadius(8)

            if ble.connectedPeripheral == nil {
                Text("Select ECG Device")
                    .font(.headline)

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
                .frame(height: 200)
            }

            HStack(spacing: 16) {
                Button(isProcessing ? "Stop" : "Start") {
                    isProcessing ? stopStream() : startStream()
                }
                .buttonStyle(.borderedProminent)
                .tint(isProcessing ? .red : .green)

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
            }


        }
        .padding()
        
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
            stopStream()
            ble.reset()
        }
        
        
        .alert("Invalid Device", isPresented: $showInvalidDeviceAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("❌ Please choose a valid ECG monitor device")
        }
        
        
        
    }

    private func startStream() {
        samples = []
        recordingBuffer = []
        isProcessing = true
        timer = Timer.publish(every: updateInterval, on: .main, in: .common)
            .autoconnect()
            .sink { _ in tick() }
        
        ble.startScanWithTimeoutAfterConnect {
            showInvalidDeviceAlert = true
        }

        
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
        
        print("💾 Saving with fs = \(fs)")

        
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
        } catch {
            print("❌ Error saving: \(error)")
        }
    }


    
}

