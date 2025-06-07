import SwiftUI
import Charts
import Combine

private let signalGain: Float = 200.0  // µV/LSB, typowo 200
private let leadName = "II"
private let resolutionBits = 16


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

                            try saveRecordingToAllFiles(baseName: baseName, fs: fs, start: start, end: end)
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

    // MARK: - Save functions
    
    
    private func saveRecordingToAllFiles(baseName: String, fs: Int, start: Date, end: Date) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try saveMatFile(to: dir.appendingPathComponent("\(baseName).mat"))
        try saveDatFile(to: dir.appendingPathComponent("\(baseName).dat"), gain: signalGain)
        try saveHeaFile(to: dir.appendingPathComponent("\(baseName).hea"), baseName: baseName, fs: fs, gain: signalGain, start: start, end: end)
        try saveRecordingAsJSON(baseName: baseName, fs: fs, start: start, end: end)  // 👈 dodaj ten wiersz
    }

    
    
    private func saveDatFile(to url: URL, gain: Float) throws {
        let int16Signal = recordingBuffer.map { Int16($0 * gain) }
        var datData = Data()
        for s in int16Signal {
            var le = s.littleEndian
            datData.append(Data(bytes: &le, count: 2))
        }
        try datData.write(to: url)
    }
    
    
    private func saveMatFile(to url: URL) throws {
        let val = [recordingBuffer]  // [[Float]]
        let dict: [String: Any] = ["val": val]
        let data = try PropertyListSerialization.data(fromPropertyList: dict, format: .binary, options: 0)
        try data.write(to: url)
    }

    
    private func saveHeaFile(to url: URL, baseName: String, fs: Int, gain: Float, start: Date, end: Date) throws {
        let settings = AppSettings.shared
        let nSamples = recordingBuffer.count
        let bitRes = resolutionBits
        let format = 16
        let baseline = 0
        let adcRes = resolutionBits
        let adcZero = 0
        let initVal = 0
        let checksum = 0
        let blockSize = 0
        let sex = settings.userSex == 0 ? "M" : "F"
        let age = settings.userAge

        let durationSec = Double(nSamples) / Double(fs)
        let recordingTime = ISO8601DateFormatter().string(from: start)

        let mainLine = "\(baseName) 1 \(fs) \(nSamples) \(bitRes)"
        let signalLine = "\(baseName).dat \(format) \(Int(gain))/mV \(baseline) 0 \(adcRes) \(adcZero) \(initVal) \(checksum) \(blockSize) \(leadName)"
        
        let commentLines = [
            "# age: \(age)",
            "# sex: \(sex)",
            "# duration: \(Int(durationSec)) seconds",
            "# start_time: \(recordingTime)",
            "# Recorded via ECG mobile app"
        ]

        let hea = ([mainLine, signalLine] + commentLines).joined(separator: "\n")
        try hea.write(to: url, atomically: true, encoding: .utf8)
    }




    private func saveRecordingAsJSON(baseName: String, fs: Int, start: Date, end: Date) throws {
        let jsonObject: [String: Any] = [
            "fs": fs,
            "lead": "II",
            "start_time": ISO8601DateFormatter().string(from: start),
            "end_time": ISO8601DateFormatter().string(from: end),
            "signal": recordingBuffer
        ]
        let data = try JSONSerialization.data(withJSONObject: jsonObject, options: .prettyPrinted)
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("\(baseName).json")
        try data.write(to: url)
        print("✅ JSON saved to \(url)")
    }


    // MARK: - Delete all files

    func deleteRecording(_ rec: ECGRecordingSet) {
        for url in [rec.json, rec.wfdbDat, rec.wfdbHea, rec.mat] {
            if FileManager.default.fileExists(atPath: url.path) {
                try? FileManager.default.removeItem(at: url)
                print("🗑️ Deleted \(url.lastPathComponent)")
            }
        }
    }
}

