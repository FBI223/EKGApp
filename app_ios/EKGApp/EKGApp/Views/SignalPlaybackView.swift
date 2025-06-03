import SwiftUI
import Charts

struct SignalPlaybackView: View {
    let filename: String

    @State private var signal: [Float] = []
    @State private var fs: Int = 128
    @State private var offset: Int = 0
    @State private var windowSize: Int = 700
    @State private var yScale: Float = 3.0
    @State private var lead: String = "II"

    @State private var startTime: Date?
    @State private var endTime: Date?
    @State private var durationSeconds: Int?

    var body: some View {
        VStack(spacing: 16) {
            // === Informacje nagłówkowe ===
            VStack(spacing: 6) {
                Text(filename)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)

                if let start = startTime, let end = endTime {
                    Text("📅 \(start.formatted(date: .abbreviated, time: .standard)) → \(end.formatted(date: .abbreviated, time: .standard))")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }

                Text("📈 FS: \(fs) Hz, Samples: \(signal.count), Lead: \(lead)")
                    .font(.subheadline)
                    .foregroundColor(.gray)

                if let dur = durationSeconds {
                    Text("⏱ Duration: \(dur) sec")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
            }

            // === Wykres ===
            Chart {
                let end = min(offset + windowSize, signal.count)
                let visible = Array(signal[offset..<end])
                ForEach(Array(visible.enumerated()), id: \.offset) { i, v in
                    LineMark(x: .value("Index", offset + i), y: .value("Value", v))
                }
            }
            .chartXScale(domain: Double(offset)...Double(offset + windowSize))
            .chartYScale(domain: -Double(yScale)...Double(yScale))
            .frame(maxHeight: 350)
            .background(Color.black.opacity(0.05))
            .cornerRadius(10)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        let dragAmount = Int(value.translation.width / 4)
                        let newOffset = offset - dragAmount
                        offset = min(max(0, newOffset), max(0, signal.count - windowSize))
                    }
            )

            // === Suwaki ===
            VStack(alignment: .leading) {
                Text("🧭 Navigate")
                    .font(.subheadline)

                HStack {
                    Text("Window: \(windowSize)")
                    Slider(value: Binding(
                        get: { Double(windowSize) },
                        set: { windowSize = Int($0) }
                    ), in: 100...5000, step: 100)
                }

                HStack {
                    Text("Y ±\(String(format: "%.1f", yScale))")
                    Slider(value: $yScale, in: 0.5...10.0, step: 0.5)
                }

                if signal.count > windowSize {
                    HStack {
                        Text("Offset: \(offset)")
                        Slider(value: Binding(
                            get: { Double(offset) },
                            set: { newValue in
                                offset = min(Int(newValue), max(0, signal.count - windowSize))
                            }
                        ), in: 0...Double(signal.count - windowSize), step: 1)
                    }
                }
            }

            Spacer()
        }
        .padding()
        .onAppear(perform: load)
    }

    private func load() {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
        do {
            let data = try Data(contentsOf: url)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let rawFs = json["fs"] as? Int,
               let rawSignal = json["signal"] as? [Double] {

                fs = rawFs
                signal = rawSignal.map { Float($0) }

                if let rawLead = json["lead"] as? String {
                    lead = rawLead
                }

                let isoFormatter = ISO8601DateFormatter()
                if let startStr = json["start_time"] as? String,
                   let endStr = json["end_time"] as? String,
                   let start = isoFormatter.date(from: startStr),
                   let end = isoFormatter.date(from: endStr) {
                    startTime = start
                    endTime = end
                    durationSeconds = Int(end.timeIntervalSince(start))
                } else {
                    durationSeconds = Int(Double(signal.count) / Double(fs))
                }
            }
        } catch {
            print("❌ Failed to load file \(filename): \(error)")
        }
    }
}

