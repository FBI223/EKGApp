import SwiftUI
import Charts

/// Lets the user pick a bundled signal (SimulationSignals/sim_*.json) and open any analysis screen
/// against a simulated ECG device instead of a real Bluetooth sensor.
struct SimulationView: View {
    @ObservedObject private var settings = AppSettings.shared

    @State private var signals: [SimulatedSignal] = []
    @State private var selectedID: String?
    @State private var isLoading = true

    private let previewSeconds = 5

    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }
    var chartColor: Color { settings.darkModeEnabled ? .cyan : .blue }

    private var selected: SimulatedSignal? {
        signals.first { $0.id == selectedID }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Choose a signal. It will be streamed in a loop at its own sample rate, exactly like a connected BLE ECG device.")
                    .font(.footnote)
                    .foregroundColor(.secondary)

                if isLoading {
                    ProgressView("Loading signals…")
                        .frame(maxWidth: .infinity)
                } else if signals.isEmpty {
                    Text("❌ No simulation signals found (expected sim_*.json in SimulationSignals).")
                        .foregroundColor(.red)
                } else {
                    signalList
                    if let signal = selected {
                        preview(of: signal)
                        analysisLinks(for: signal)
                    }
                }
            }
            .padding()
        }
        .background(backgroundColor.ignoresSafeArea())
        .navigationTitle("🧪 Simulation")
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
        .task {
            guard signals.isEmpty else { return }
            let loaded = await Task.detached(priority: .userInitiated) {
                SimulatedSignalLibrary.loadAll()
            }.value
            signals = loaded
            selectedID = selectedID ?? loaded.first?.id
            isLoading = false
        }
    }

    // MARK: - Subviews

    private var signalList: some View {
        VStack(spacing: 8) {
            ForEach(signals) { signal in
                Button {
                    selectedID = signal.id
                } label: {
                    HStack {
                        Image(systemName: signal.id == selectedID ? "largecircle.fill.circle" : "circle")
                            .foregroundColor(.red)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(signal.name)
                                .font(.headline)
                                .foregroundColor(foregroundColor)
                            Text("\(signal.fs) Hz · \(formatDuration(signal.durationSeconds))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            if !signal.description.isEmpty {
                                Text(signal.description)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.leading)
                            }
                        }
                        Spacer()
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(signal.id == selectedID ? Color.red : Color.gray.opacity(0.4), lineWidth: 1.5)
                    )
                }
            }
        }
    }

    private func preview(of signal: SimulatedSignal) -> some View {
        let count = min(signal.samples.count, previewSeconds * signal.fs)
        let points = Array(signal.samples.prefix(count))
        return VStack(alignment: .leading, spacing: 4) {
            Text("Preview (first \(previewSeconds) s)")
                .font(.caption)
                .foregroundColor(.secondary)
            Chart {
                ForEach(0..<points.count, id: \.self) { i in
                    LineMark(
                        x: .value("Time", Double(i) / Double(signal.fs)),
                        y: .value("mV", points[i])
                    )
                    .foregroundStyle(chartColor)
                }
            }
            .chartXAxisLabel("s")
            .chartYAxisLabel("mV")
            .frame(height: 160)
        }
    }

    private func analysisLinks(for signal: SimulatedSignal) -> some View {
        VStack(spacing: 10) {
            if settings.sampleRateIn != signal.fs {
                Text("ℹ️ Sample Rate In will be set to \(signal.fs) Hz when the simulation starts.")
                    .font(.caption)
                    .foregroundColor(.orange)
            }
            link("📈 Beat Analysis") { BeatAnalysisView(simulatedSignal: signal) }
            link("📈 Rhythm Analysis") { RhythmAnalysisView(simulatedSignal: signal) }
            link("📈 Wave Analysis") { WaveAnalysisView(simulatedSignal: signal) }
            link("📡 Signal Recorder") { SignalRecorderView(simulatedSignal: signal) }
        }
    }

    private func link<Destination: View>(_ title: String, @ViewBuilder destination: @escaping () -> Destination) -> some View {
        NavigationLink {
            destination()
        } label: {
            Text(title)
                .font(.title3)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                .foregroundColor(.white)
                .cornerRadius(14)
        }
    }

    private func formatDuration(_ seconds: Int) -> String {
        String(format: "%d:%02d min", seconds / 60, seconds % 60)
    }
}
