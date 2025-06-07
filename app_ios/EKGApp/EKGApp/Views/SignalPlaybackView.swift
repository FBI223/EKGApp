import SwiftUI
import Charts

struct SignalPlaybackView: View {
    let filename: String

    @State private var signals: [[Float]] = []
    @State private var leads: [String] = []
    @State private var selectedLeadIndex: Int = 0
    private var currentSignal: [Float] {
        guard selectedLeadIndex < signals.count else { return [] }
        return signals[selectedLeadIndex]
    }
    
    @State private var fs: Int = 128
    @State private var offset: Int = 0
    @State private var windowSize: Int = 700
    @State private var yScale: Float = 3.0

    @State private var startTime: Date?
    @State private var endTime: Date?
    @State private var durationSeconds: Int?
    @State private var lastScale: CGFloat = 1.0


    
    
    func detectECGFormat(url: URL) -> ECGFormat? {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "json": return .json
        case "hea": return .wfdb
        default: return nil
        }
    }

    
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

                Text("📈 FS: \(fs) Hz, Samples: \(currentSignal.count), Lead: \(leads.indices.contains(selectedLeadIndex) ? leads[selectedLeadIndex] : "—")")
                    .font(.subheadline)
                    .foregroundColor(.gray)

                if let dur = durationSeconds {
                    Text("⏱ Duration: \(dur) sec")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }

                if leads.count > 1 {
                    Picker("Lead", selection: $selectedLeadIndex) {
                        ForEach(leads.indices, id: \.self) { i in
                            Text(leads[i]).tag(i)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                }
            }

            // === Wykres ===
            Chart {
                let sig = currentSignal
                let end = min(offset + windowSize, sig.count)
                let visible = Array(sig[offset..<end])
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

                        offset = min(offset, max(0, currentSignal.count - windowSize))
                    }
                    .onEnded { _ in lastScale = 1.0 }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { value in
                        let dragAmount = Int(value.translation.width / 4)
                        let newOffset = offset - dragAmount
                        offset = min(max(0, newOffset), max(0, currentSignal.count - windowSize))
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

                Text("🧭 Navigate").font(.subheadline).bold()

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
                    Text("Offset: \(offset)").opacity(currentSignal.count > windowSize ? 1 : 0.3)
                    Slider(value: Binding(
                        get: { Double(offset) },
                        set: { newValue in
                            offset = min(Int(newValue), max(0, currentSignal.count - windowSize))
                        }
                    ), in: 0...Double(max(1, currentSignal.count - windowSize)), step: 1)
                    .disabled(currentSignal.count <= windowSize)
                    .opacity(currentSignal.count > windowSize ? 1 : 0.3)
                }
            }
            .frame(maxWidth: 600, alignment: .leading)
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
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fullURL = docDir.appendingPathComponent(filename)
        let ext = fullURL.pathExtension.lowercased()

        // === Rozpoznaj format ===
        let format: ECGFormat
        switch ext {
        case "json": format = .json
        case "hea":  format = .wfdb
        default:
            print("❌ Unsupported file extension: \(ext)")
            return
        }

        // === Wczytaj przez unified API ===
        guard let ecg = ECGLoader.loadMultiLead(from: fullURL, format: format) else {
            print("❌ Could not load ECG recording: \(filename)")
            return
        }


        // === Metadane
        fs = ecg.fs
        startTime = ecg.startTime
        endTime = ecg.endTime
        durationSeconds = ecg.endTime.flatMap { end in
            ecg.startTime.map { Int(end.timeIntervalSince($0)) }
        } ?? Int(Double(ecg.signals.first?.count ?? 0) / Double(fs))

        // === Leads
        leads = ecg.leads
        signals = ecg.signals

        // === Wybierz kanał II albo pierwszy
        if let iiIndex = leads.firstIndex(where: { $0.uppercased() == "II" }) {
            selectedLeadIndex = iiIndex
        } else {
            selectedLeadIndex = 0
        }

        // === Okno
        let len = currentSignal.count
        windowSize = max(min(fs, len), min(700, len))
    }

    

}

