

import SwiftUI
import Charts
import Combine




class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var samplesPerTick: Int = 10
    @Published var updateInterval: Float = 0.05

    @Published var windowLengthRythm: Int = 1280
    @Published var windowLengthBeat: Int = 192
    @Published var windowLengthRythmResampled: Int = 3600
    @Published var windowLengthBeatResampled: Int = 540

    @Published var analysisWindowSeconds: Float = 1.5
    @Published var darkModeEnabled: Bool = true
    @Published var showDebugInfo: Bool = true
    @Published var sampleRateIn: Int = 128
    @Published var yAxisRange: ClosedRange<Float> = -3...3
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



