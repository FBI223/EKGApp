

import SwiftUI
import Charts
import Combine




class AppSettings: ObservableObject {
    static let shared = AppSettings()

    
    @Published var samplesPerTick: Int = 10
    @Published var updateInterval: Float = 0.05      // 50 ms
    
    
    @Published var windowLengthRythm: Int = 1280
    @Published var windowLengthBeat: Int = 192
    @Published var windowLengthRythmResampled: Int = 3600
    @Published var windowLengthBeatResampled: Int = 540
    
    @Published var analysisWindowSeconds: Float = 1.5
    @Published var darkModeEnabled: Bool = false
    @Published var showDebugInfo: Bool = true
    @Published var sampleRateIn: Int = 128
    @Published var sampleRateOut: Int = 360
    @Published var maxBufferSize: Int = 5000
    @Published var yAxisRange: ClosedRange<Float> = -3...3
}






struct SettingsView: View {
    @ObservedObject private var settings = AppSettings.shared
    
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
                Picker("Input Sample Rate", selection: $settings.sampleRateIn) {
                    Text("128 Hz").tag(128)
                    Text("250 Hz").tag(250)
                    Text("360 Hz").tag(360)
                }
                .pickerStyle(.segmented)

                Picker("Output Sample Rate", selection: $settings.sampleRateOut) {
                    Text("128 Hz").tag(128)
                    Text("360 Hz").tag(360)
                    Text("500 Hz").tag(500)
                }
                .pickerStyle(.segmented)

                Stepper(value: $settings.maxBufferSize, in: 1000...20000, step: 1000) {
                    Text("Max Buffer: \(settings.maxBufferSize) samples")
                }

                VStack(alignment: .leading) {
                    Text("Analysis Window: \(settings.analysisWindowSeconds, specifier: "%.1f") sec")
                    Slider(value: $settings.analysisWindowSeconds, in: 1.0...3.0, step: 0.1)
                }
                .padding(.top, 4)
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
