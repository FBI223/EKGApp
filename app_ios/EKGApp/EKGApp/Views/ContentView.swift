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
            VStack(spacing: 12) {
                // Górna ikona i tytuł
                VStack(spacing: 6) {
                    Image(systemName: "waveform.path.ecg.rectangle")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 40, height: 40)
                        .foregroundColor(.red)
                    Text("EKG App")
                        .font(.title)
                        .fontWeight(.semibold)
                        .foregroundColor(foregroundColor)
                }
                .padding(.top, 24)

                // Przyciski
                VStack(spacing: 12) {
                    NavigationLink(destination: BeatAnalysisView()) {
                        Text("📈 Beat Analysis")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: RhythmAnalysisView()) {
                        Text("📈 Rhythm Analysis")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: WaveAnalysisView()) {
                        Text("📈 Wave Analysis")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: SignalRecorderView()) {
                        Text("📡 Signal Recorder")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: SimulationView()) {
                        Text("🧪 Simulation")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: SignalBrowserView()) {
                        Text("📁 Record Browser")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: KnowledgeView()) {
                        Text("📚 Knowledge")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: InfoView()) {
                        Text("ℹ️ Label Info")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }

                    NavigationLink(destination: SettingsView()) {
                        Text("⚙️ Settings")
                            .font(.title3)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(14)
                    }
                }

                Text("⚠️ This app is not intended for medical diagnosis. If you experience any health problems, please consult a doctor!")
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.gray)
                    .padding(.horizontal)

                Spacer()
            }
            .padding()
            .background(backgroundColor.ignoresSafeArea())
        }
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
    }
}

