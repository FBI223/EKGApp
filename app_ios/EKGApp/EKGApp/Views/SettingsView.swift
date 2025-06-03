

import SwiftUI
import Charts
import Combine


class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var samplesPerTick: Int = 10 {
        didSet { print("samplesPerTick changed from \(oldValue) to \(samplesPerTick)") }
    }

    @Published var updateInterval: Float = 0.05 {
        didSet { print("updateInterval changed from \(oldValue) to \(updateInterval)") }
    }

    @Published var windowLengthRythm: Int = 1280 {
        didSet { print("windowLengthRythm changed from \(oldValue) to \(windowLengthRythm)") }
    }

    @Published var windowLengthBeat: Int = 192 {
        didSet { print("windowLengthBeat changed from \(oldValue) to \(windowLengthBeat)") }
    }

    @Published var windowLengthRythmResampled: Int = 3600 {
        didSet { print("windowLengthRythmResampled changed from \(oldValue) to \(windowLengthRythmResampled)") }
    }

    @Published var windowLengthBeatResampled: Int = 540 {
        didSet { print("windowLengthBeatResampled changed from \(oldValue) to \(windowLengthBeatResampled)") }
    }

    @Published var analysisWindowSeconds: Float = 1.5 {
        didSet { print("analysisWindowSeconds changed from \(oldValue) to \(analysisWindowSeconds)") }
    }

    @Published var darkModeEnabled: Bool = true {
        didSet { print("darkModeEnabled changed from \(oldValue) to \(darkModeEnabled)") }
    }

    @Published var showDebugInfo: Bool = true {
        didSet { print("showDebugInfo changed from \(oldValue) to \(showDebugInfo)") }
    }

    @Published var sampleRateIn: Int = 128 {
        didSet { print("sampleRateIn changed from \(oldValue) to \(sampleRateIn)") }
    }

    @Published var yAxisRange: ClosedRange<Float> = -3...3 {
        didSet { print("yAxisRange changed from \(oldValue) to \(yAxisRange)") }
    }

    @Published var userAge: Int = 25 {
        didSet { print("userAge changed from \(oldValue) to \(userAge)") }
    }

    @Published var userSex: Int = 1 {  // 0 = Male, 1 = Female
        didSet { print("userSex changed from \(oldValue) to \(userSex)") }
    }
}








struct SettingsView: View {
    @ObservedObject private var settings = AppSettings.shared

    @State private var inputRateText: String
    @State private var rateError: String? = nil

    init() {
        _inputRateText = State(initialValue: "\(AppSettings.shared.sampleRateIn)")
    }

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }
    var accentColor: Color { settings.darkModeEnabled ? .cyan : .blue }

    var body: some View {
        
        Form {
            // Display
            Section(header: Text("🌙 Display").foregroundColor(accentColor)) {
                Toggle("Dark Mode", isOn: $settings.darkModeEnabled)
                Toggle("Show Debug Info", isOn: $settings.showDebugInfo)
            }
            
            
            
            
            // Demographics
            Section(header: Text("🧍 Demographics").foregroundColor(accentColor)) {
                Picker("Sex", selection: $settings.userSex) {
                    Text("Male").tag(0)
                    Text("Female").tag(1)
                }
                .pickerStyle(.segmented)

                VStack(alignment: .leading) {
                    Text("Age: \(settings.userAge)")
                        .font(.headline)

                    Picker("Age", selection: $settings.userAge) {
                        ForEach(1...110, id: \.self) { age in
                            Text("\(age)").tag(age)
                        }
                    }
                    .pickerStyle(.wheel)
                    .frame(height: 150)
                    .clipped()
                }
            }
            
            // Processing
        
            Section(header: Text("⚙️ Processing Settings").foregroundColor(accentColor)) {
                VStack(alignment: .leading) {
                    Text("Sample Rate In: \(settings.sampleRateIn) Hz")
                        .font(.headline)

                    Picker("Sample Rate", selection: $settings.sampleRateIn) {
                        ForEach(100...500, id: \.self) { rate in
                            Text("\(rate) Hz").tag(rate)
                        }
                    }
                    .pickerStyle(.wheel)
                    .frame(height: 150) // wysokość widocznego pokrętła
                    .clipped()
                }


            }

            // Chart
            Section(header: Text("📊 Chart Settings").foregroundColor(accentColor)) {
                VStack(alignment: .leading) {
                    Text("Y-Axis Range: ±\(Int(settings.yAxisRange.upperBound))")
                    Slider(
                        value: Binding(
                            get: { settings.yAxisRange.upperBound },
                            set: { settings.yAxisRange = -$0...$0 }
                        ),
                        in: 1...10,
                        step: 0.5
                    )
                }
                .padding(.top, 4)
            }
        }
        .navigationTitle("Settings")
        .background(backgroundColor)
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
    }
}



