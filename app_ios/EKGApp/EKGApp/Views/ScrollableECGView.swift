import SwiftUI
import Charts
import Combine

struct ScrollableECGView: View {
    @StateObject private var ble = EKGBLEManager()
    @ObservedObject private var settings = AppSettings.shared
    
    @State private var allSamples: [Float] = []
    @State private var liveMode = true
    @State private var scrollOffset: CGFloat = 0.0

    let defaultWindowSeconds: Int = 7

    var visibleSampleCount: Int {
        Int(settings.sampleRateIn * defaultWindowSeconds)
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text(liveMode ? "📡 Live Mode" : "🕓 History Mode")
                    .font(.headline)
                Spacer()
                if !liveMode {
                    Button("Back to Live") {
                        withAnimation {
                            liveMode = true
                            scrollOffset = 0
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            GeometryReader { geo in
                let visible = liveMode
                    ? Array(allSamples.suffix(visibleSampleCount))
                    : Array(allSamples.dropLast(visibleSampleCount).suffix(visibleSampleCount))

                Chart {
                    ForEach(0..<visible.count, id: \.self) { i in
                        LineMark(
                            x: .value("Sample", i),
                            y: .value("Value", visible[i])
                        )
                    }
                }
                .chartXScale(domain: 0...visibleSampleCount)
                .chartYScale(domain: Double(settings.yAxisRange.lowerBound)...Double(settings.yAxisRange.upperBound))
                .gesture(
                    DragGesture()
                        .onChanged { _ in
                            if liveMode && allSamples.count > visibleSampleCount {
                                liveMode = false
                            }
                        }
                )
                .frame(height: 250)
            }
            .padding(.horizontal)

            Spacer()
        }
        .padding()
        .navigationTitle("🖐️ Scrollable ECG")
        .onAppear {
            allSamples = []
        }
        .onReceive(Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()) { _ in
            if !ble.rawBuffer.isEmpty {
                let new = ble.rawBuffer.removeFirst()
                allSamples.append(new)
                if allSamples.count > settings.maxBufferSize {
                    allSamples.removeFirst(allSamples.count - settings.maxBufferSize)
                }
                if liveMode {
                    scrollOffset = 0
                }
            }
        }
    }
}

