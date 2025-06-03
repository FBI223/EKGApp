import SwiftUI

struct KnowledgeArticle: Identifiable {
    let id = UUID()
    let title: String
    let content: String
}

struct KnowledgeView: View {
    let articles: [KnowledgeArticle] = [
        KnowledgeArticle(
            title: "Cardiac Conduction System",
            content: "The cardiac conduction system consists of the SA node, AV node, bundle of His, bundle branches, and Purkinje fibers. It is responsible for initiating and coordinating the heartbeat. The SA node generates impulses, causing atrial contraction, and the signal is delayed at the AV node before passing through the ventricles."
        ),
        KnowledgeArticle(
            title: "ECG Electrodes and Leads",
            content: "Electrodes are sensors placed on the skin to detect electrical activity of the heart. Leads are combinations of these electrodes that form the views on an ECG. Lead II, commonly used in monitoring, records the voltage difference between the right arm and left leg."
        ),
        KnowledgeArticle(
            title: "Exercise and Heart Health",
            content: "Regular aerobic exercise improves cardiovascular health, strengthens the heart muscle, and can reduce the risk of arrhythmias. Exercise enhances autonomic tone and can reduce resting heart rate. Always consult a doctor before starting new physical activity if you have a heart condition."
        ),
        KnowledgeArticle(
            title: "Ablation Procedures",
            content: "Catheter ablation is used to treat abnormal heart rhythms. It involves threading a catheter through blood vessels to the heart and delivering energy (radiofrequency or cryoablation) to destroy tissue causing arrhythmia. It is effective in treating atrial fibrillation, SVT, and other arrhythmias."
        ),
        KnowledgeArticle(
            title: "Pacemakers and ICDs",
            content: "Pacemakers are devices implanted under the skin to regulate slow heart rhythms. Implantable cardioverter-defibrillators (ICDs) monitor heart rhythms and deliver shocks to correct dangerous tachycardias or fibrillation. These devices are essential in patients at risk of sudden cardiac death."
        ),
        KnowledgeArticle(
            title: "Blood Groups and Compatibility",
            content: "Human blood is classified into A, B, AB, and O types, and Rh-positive or negative. Blood group compatibility is crucial for transfusions. O-negative is the universal donor, while AB-positive is the universal recipient. Matching reduces risk of immune reactions."
        )
    ]

    var body: some View {
        List(articles) { article in
            NavigationLink(destination: KnowledgeDetailView(article: article)) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(article.title)
                        .font(.title3) // większa czcionka
                        .foregroundColor(.primary)
                        .padding(.vertical, 8)
                }
            }
        }
        .navigationTitle("📚 Knowledge Base")
        .listStyle(.insetGrouped)
    }
}

struct KnowledgeDetailView: View {
    let article: KnowledgeArticle

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 32) {
                Text(article.title)
                    .font(.title2)
                    .bold()

                Text(article.content)
                    .font(.system(size: 24)) // <-- większa czcionka tylko dla treści
            }
            .padding()
        }
        .navigationTitle(article.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}


