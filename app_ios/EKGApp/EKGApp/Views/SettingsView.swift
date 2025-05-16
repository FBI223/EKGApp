

import SwiftUI
import Charts
import Combine



class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var darkModeEnabled: Bool = false
    @Published var showDebugInfo: Bool = true
    @Published var sampleRate: Int = 128 // Can be 128 or 360
    @Published var yAxisRange: ClosedRange<Double> = -3...3
    @Published var maxBufferSize: Int = 5000
}




struct SettingsView: View {
    @ObservedObject private var settings = AppSettings.shared

    var body: some View {
        Form {
            Section(header: Text("Display")) {
                Toggle("Dark Mode", isOn: $settings.darkModeEnabled)
                Toggle("Show Debug Info", isOn: $settings.showDebugInfo)
            }

            Section(header: Text("Processing")) {
                Picker("Sample Rate", selection: $settings.sampleRate) {
                    Text("128 Hz").tag(128)
                    Text("360 Hz").tag(360)
                }
                Stepper(value: $settings.maxBufferSize, in: 1000...20000, step: 1000) {
                    Text("Max Buffer: \(settings.maxBufferSize) samples")
                }
            }

            Section(header: Text("Chart")) {
                Slider(value: Binding(
                    get: { settings.yAxisRange.upperBound },
                    set: { settings.yAxisRange = settings.yAxisRange.lowerBound...$0 }
                ), in: 1...10, step: 0.5) {
                    Text("Y-Axis ±\(Int(settings.yAxisRange.upperBound))")
                }
            }
        }
        .navigationTitle("Settings")
    }
}
