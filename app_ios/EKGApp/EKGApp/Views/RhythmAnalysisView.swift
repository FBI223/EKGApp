import SwiftUI
import Charts
import Combine

struct RhythmAnalysisView: View {
    @StateObject private var ble = EKGBLEManager()

    @State private var rhythmBuffer: [Float] = []
    @State private var prediction: String = "—"
    @State private var rhythmClassCounts: [String: Int] = [
        "NSR": 0,
        "AF_FLUTTER": 0,
        "PAC": 0,
        "PVC": 0,
        "BBB": 0,
        "SVT": 0,
        "AV_BLOCK": 0,
        "TORSADES": 0
    ]

    @State private var timer: AnyCancellable?
    @State private var isProcessing = false

    @Environment(\.dismiss) var dismiss
    @ObservedObject private var settings = AppSettings.shared

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }
    var chartColor: Color { settings.darkModeEnabled ? .cyan : .blue }

    let windowLength = 1280
    private let model = RhythmClassifier()


    
   
    
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
                            Text("📊 Rhythm Prediction")
                                .font(.title2)
                                .bold()
                                .foregroundColor(foregroundColor)

                            Text(prediction)
                                .font(.system(size: 32, weight: .semibold, design: .rounded))
                                .foregroundColor(.orange)

                            Chart {
                                let visible = Array(rhythmBuffer.suffix(windowLength))
                                ForEach(0..<visible.count, id: \.self) { i in
                                    LineMark(
                                        x: .value("Index", i),
                                        y: .value("Voltage", visible[i])
                                    )
                                    .foregroundStyle(chartColor)
                                }
                            }
                            .chartXScale(domain: 0...windowLength)
                            .chartYScale(domain: Double(settings.yAxisRange.lowerBound)...Double(settings.yAxisRange.upperBound))
                            .frame(height: geometry.size.height * 0.35)
                            .background(backgroundColor)
                            .cornerRadius(10)
                            .padding(.horizontal)

                            HStack {
                                ForEach( ["NSR", "AF_FLUTTER", "PAC", "PVC", "BBB", "SVT", "AV_BLOCK", "TORSADES"] , id: \.self) { key in
                                    VStack {
                                        Text(key)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text("\(rhythmClassCounts[key] ?? 0)")
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
                                    rhythmBuffer = []
                                    prediction = "—"
                                    rhythmClassCounts =  ["NSR" : 0, "AF_FLUTTER" : 0 , "PAC" : 0, "PVC" : 0, "BBB" : 0, "SVT" : 0, "AV_BLOCK" : 0, "TORSADES" : 0]
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
            ble.disconnect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                ble.startScan()
            }
        }
        .onDisappear {
            stopProcessing()
            ble.reset()
        }
    }

    
    
    func startProcessing() {
        rhythmBuffer = []
        prediction = "—"
        rhythmClassCounts =  ["NSR" : 0, "AF_FLUTTER" : 0 , "PAC" : 0, "PVC" : 0, "BBB" : 0, "SVT" : 0, "AV_BLOCK" : 0, "TORSADES" : 0]
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
            let ecg1280 = Array(rhythmBuffer.suffix(windowLength))
            let age = settings.userAge      // dodaj to w AppSettings jeśli nie masz
            let sex = settings.userSex      // 0.0 = mężczyzna, 1.0 = kobieta

            let predictedClass = model.predict(ecgInput: ecg1280, age: Float(age), sex: Float(sex))
            prediction = predictedClass
            rhythmClassCounts[predictedClass, default: 0] += 1

            print("[Rhythm] 10s classified as \(predictedClass)")
            rhythmBuffer.removeFirst(windowLength)
        }

    }
}




