import SwiftUI
import Charts
import Combine

struct ChartDataPoint: Identifiable {
    let id = UUID()
    let x: Int
    let y: Float
}

struct SignalMeta: Codable {
    let id: Int
    let fs: Int
    let gain: Int
    let baseline: Int
    let source: String
    let lead: String
    let name: String
}



struct ContentView: View {
    @StateObject private var ble = EKGBLEManager()
    @State private var samples: [Float] = Array(repeating: 0, count: 500)
    @State private var recordingBuffer: [Float] = []
    @State private var currentIndex = 0
    @State private var timer: AnyCancellable?
    @State private var isProcessing = false
    @State private var isRecording = false
    @State private var showingMetaInfo = false


    let maxVisibleSamples = 500
    let updateInterval = 0.05
    let samplesPerTick = 10

    var body: some View {
        VStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("META: \(ble.latestMetaMessage ?? "No META received yet")")
                    .font(.caption)
                    .foregroundColor(.blue)
                    .padding(5)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            .padding(.horizontal)


            Divider()

            if ble.connectedPeripheral == nil {
                List(ble.devices, id: \ .identifier) { device in
                    Button(action: { ble.connect(to: device) }) {
                        Text(device.name ?? "Unknown")
                    }
                }
            } else {
                Chart(liveSamples) { point in
                    LineMark(x: .value("Index", point.x), y: .value("Voltage", point.y))
                }
                .chartYScale(domain: -3...3)
                .frame(height: 250)
                .padding()


                HStack {
                    Button(isProcessing ? "Stop" : "Start") {
                        isProcessing ? stopProcessing() : startProcessing()
                    }
                    .padding()
                    .background(isProcessing ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Button(isRecording ? "Stop Rec" : "Record") {
                        isRecording ? stopRecording() : startRecording()
                    }
                    .padding()
                    .background(isRecording ? Color.orange : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Button("Disconnect") {
                        ble.disconnect()
                        stopProcessing()
                        samples = Array(repeating: 0, count: maxVisibleSamples)
                    }
                    .padding()
                    .background(Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Button("Status") {
                        showingMetaInfo = true
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .padding(.top)

                
                
                
            }
            
            

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(ble.eventLogs.reversed(), id: \ .self) { log in
                        Text(log).font(.system(size: 10)).foregroundColor(.secondary)
                    }
                }
                .padding()
            }
            .frame(maxHeight: 200)
            .background(Color.black.opacity(0.05))
            .cornerRadius(8)
        }
        .onAppear {
            ble.startScan()
        }

        .sheet(isPresented: $showingMetaInfo) {
            VStack(alignment: .leading, spacing: 10) {
                Text("META Information")
                    .font(.title2)
                    .padding(.bottom, 10)
                
                Group {
                    Text("Name: \(ble.meta.name)")
                    Text("Lead: \(ble.meta.lead)")
                    Text("Source: \(ble.meta.source)")
                    Text("Fs: \(ble.meta.fs) Hz")
                    Text("Gain: \(ble.meta.gain)")
                    Text("Baseline: \(ble.meta.baseline)")
                    Text("ID: \(ble.meta.id)")
                }
                .font(.body)
                
                Spacer()
                
                Button("Close") {
                    showingMetaInfo = false
                }
                .padding()
                .background(Color.red)
                .foregroundColor(.white)
                .cornerRadius(8)
                Spacer()
            }
            .padding()
        }
        .padding()
    }

    var liveSamples: [ChartDataPoint] {
        samples.enumerated().map { (i, v) in
            ChartDataPoint(x: i, y: v)
        }
    }

    func startProcessing() {
        samples = Array(repeating: 0, count: maxVisibleSamples)
        currentIndex = 0
        isProcessing = true

        timer = Timer.publish(every: updateInterval, on: .main, in: .common)
            .autoconnect()
            .sink { _ in
                updateSamples()
            }
    }

    func stopProcessing() {
        isProcessing = false
        timer?.cancel()
    }

    func startRecording() {
        recordingBuffer.removeAll()
        isRecording = true
    }

    func stopRecording() {
        isRecording = false
        saveRecording()
    }

    func updateSamples() {
        guard isProcessing else { return }
        guard !ble.rawBuffer.isEmpty else { return }

        for _ in 0..<samplesPerTick {
            if !ble.rawBuffer.isEmpty {
                let nextValue = ble.rawBuffer.removeFirst()

                samples[currentIndex] = nextValue
                currentIndex = (currentIndex + 1) % maxVisibleSamples

                if isRecording {
                    recordingBuffer.append(nextValue)
                }
            }
        }
    }

    func saveRecording() {
        let filename = "EKG_" + formattedDate() + ".csv"
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsURL.appendingPathComponent(filename)

        var csvText = "# fs=\(ble.meta.fs)\n# lead=\(ble.meta.lead)\n# source=\(ble.meta.source)\n"
        for sample in recordingBuffer {
            csvText += "\(sample)\n"
        }

        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("✅ Saved recording at:", fileURL)
        } catch {
            print("❌ Failed to save recording:", error)
        }
    }

    func formattedDate() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return formatter.string(from: Date())
    }


}

