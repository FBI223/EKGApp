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
    @State private var lastScale: CGFloat = 1.0

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
                MagnificationGesture()
                    .onChanged { scale in
                        let delta = scale / lastScale
                        lastScale = scale

                        if delta > 1.01 {
                            windowSize = max(100, Int(CGFloat(windowSize) / delta))
                            yScale = max(0.5, yScale / Float(delta))
                        } else if delta < 0.99 {
                            windowSize = min(5000, Int(CGFloat(windowSize) / delta))
                            yScale = min(10.0, yScale / Float(delta))
                        }
                        
                        
                        // ⛔️ Zabezpieczenie: offset nie może wyjść poza sygnał
                        offset = min(offset, max(0, signal.count - windowSize))
                    }
                    .onEnded { _ in
                        lastScale = 1.0
                    }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { value in
                        let dragAmount = Int(value.translation.width / 4)
                        let newOffset = offset - dragAmount
                        offset = min(max(0, newOffset), max(0, signal.count - windowSize))
                    }
            )

            

            // === Nawigacja + suwaki + reset ===
            VStack(alignment: .leading, spacing: 12) {
                Button(action: {
                    offset = 0
                    windowSize = 700
                    yScale = 2.0
                }) {
                    Label("Reset Chart", systemImage: "arrow.uturn.left")
                        .labelStyle(.titleAndIcon)
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)

                Text("🧭 Navigate")
                    .font(.subheadline)
                    .bold()

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

                HStack {
                    Text("Offset: \(offset)")
                        .opacity(signal.count > windowSize ? 1 : 0.3)

                    Slider(value: Binding(
                        get: { Double(offset) },
                        set: { newValue in
                            offset = min(Int(newValue), max(0, signal.count - windowSize))
                        }
                    ), in: 0...Double(max(1, signal.count - windowSize)), step: 1)
                    .disabled(signal.count <= windowSize)
                    .opacity(signal.count > windowSize ? 1 : 0.3)
                }
            }
            .frame(maxWidth: 600, alignment: .leading) // 👈 szersze okno na suwaki
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
            )
            .padding(.horizontal)



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

                guard rawSignal.count >= rawFs else {
                    print("❌ Signal too short (<1 sec). Skipping.")
                    return
                }

                fs = rawFs
                signal = rawSignal.map { Float($0) }

                // Ustaw rozsądny windowSize
                let minWindowSize = rawFs
                let defaultWindow = 700
                let maxWindowSize = 5000
                windowSize = max(minWindowSize, min(defaultWindow, rawSignal.count))

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

