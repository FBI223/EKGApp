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
            VStack(spacing: 16) {
                // Górna ikona i tytuł
                VStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg.rectangle")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 44, height: 44)
                        .foregroundColor(.red)
                    Text("EKG App")
                        .font(.largeTitle)
                        .fontWeight(.semibold)
                        .foregroundColor(foregroundColor)
                }
                .padding(.top, 32)

                // Przyciski
                VStack(spacing: 16) {
                    NavigationLink(destination: BeatAnalysisView()) {
                        Text("📈 Beat Analysis")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: RhythmAnalysisView()) {
                        Text("📈 Rhythm Analysis")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: SignalRecorderView()) {
                        Text("📡 Signal Recorder")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: SignalBrowserView()) {
                        Text("📁 Record Browser")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: KnowledgeView()) {
                        Text("📚 Knowledge")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: InfoView()) {
                        Text("ℹ️ Label Info")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: SettingsView()) {
                        Text("⚙️ Settings")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0))
                            .foregroundColor(.white)
                            .cornerRadius(16)
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

