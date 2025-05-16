import SwiftUI
import Charts
import Combine

struct ContentView: View {
    public enum AnalysisMode {
        case beat
        case rhythm
    }

    @ObservedObject private var settings = AppSettings.shared
    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                HStack(spacing: 12) {
                    Image(systemName: "bolt.horizontal.circle.fill")
                        .foregroundColor(.green)
                        .font(.title2)
                    Text("Connected to ECG device")
                        .font(.headline)
                        .foregroundColor(foregroundColor)
                }

                NavigationLink(destination: BeatAnalysisView()) {
                    Text("🔎 Beat Analysis")
                        .font(.title2)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                NavigationLink(destination: RhythmAnalysisView()) {
                    Text("📈 Rhythm Analysis")
                        .font(.title2)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                NavigationLink(destination: SettingsView()) {
                    Text("⚙️ Settings")
                        .font(.title3)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 24)
                        .background(Color.gray.opacity(0.2))
                        .foregroundColor(foregroundColor)
                        .cornerRadius(10)
                }

                Spacer()
            }
            .padding()
            .background(backgroundColor.ignoresSafeArea())
            .navigationTitle("ECG App")
        }
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
    }
}

