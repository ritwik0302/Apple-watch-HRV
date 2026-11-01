import SwiftUI
import HealthKit

@main struct RecoveryApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct Night: Identifiable {
    let id: Date
    let hrv, rhr, rr, sleep: Double
    var score: Int?
}

struct Health {
    let store = HKHealthStore()
    let cal = Calendar.current

    func quantity(_ id: HKQuantityTypeIdentifier, _ unit: HKUnit, _ from: Date) async -> [(Date, Double)] {
        let p = HKQuery.predicateForSamples(withStart: from, end: nil)
        let d = HKSampleQueryDescriptor(predicates: [.quantitySample(type: HKQuantityType(id), predicate: p)], sortDescriptors: [], limit: HKObjectQueryNoLimit)
        return ((try? await d.result(for: store)) ?? []).map { ($0.startDate, $0.quantity.doubleValue(for: unit)) }
    }

    func sleep(_ from: Date) async -> [(Date, Double)] {
        let p = HKQuery.predicateForSamples(withStart: from, end: nil)
        let d = HKSampleQueryDescriptor(predicates: [.categorySample(type: HKCategoryType(.sleepAnalysis), predicate: p)], sortDescriptors: [], limit: HKObjectQueryNoLimit)
        let asleep = HKCategoryValueSleepAnalysis.allAsleepValues.map(\.rawValue)
        return ((try? await d.result(for: store)) ?? []).filter { asleep.contains($0.value) }
            .map { ($0.startDate, $0.endDate.timeIntervalSince($0.startDate) / 3600) }
    }

    // Groups values by wake day. night = true keeps only 8pm to 10am samples.
    func bucket(_ s: [(Date, Double)], night: Bool) -> [Date: [Double]] {
        var r: [Date: [Double]] = [:]
        for (t, v) in s {
            if night {
                let x = t.addingTimeInterval(4 * 3600), day = cal.startOfDay(for: x)
                if x.timeIntervalSince(day) >= 14 * 3600 { continue }
                r[day, default: []].append(v)
            } else {
                r[cal.startOfDay(for: t), default: []].append(v)
            }
        }
        return r
    }

    func mean(_ a: [Double]) -> Double { a.reduce(0, +) / Double(a.count) }

    func nights(_ days: Int) async -> [Night] {
        let types: Set<HKObjectType> = [HKQuantityType(.heartRateVariabilitySDNN), HKQuantityType(.restingHeartRate), HKQuantityType(.respiratoryRate), HKCategoryType(.sleepAnalysis)]
        try? await store.requestAuthorization(toShare: [], read: types)
        let today = cal.startOfDay(for: Date())
        let from = cal.date(byAdding: .day, value: -(days + 1), to: today)!
        let bpm = HKUnit.count().unitDivided(by: .minute())

        async let h = quantity(.heartRateVariabilitySDNN, .secondUnit(with: .milli), from)
        async let r = quantity(.restingHeartRate, bpm, from)
        async let q = quantity(.respiratoryRate, bpm, from)
        async let sl = sleep(from)
        let hrv = bucket(await h, night: true), rhr = bucket(await r, night: false)
        let rr = bucket(await q, night: true), slp = bucket(await sl, night: true)

        var asc: [Night] = hrv.keys.compactMap { day in
            guard day >= from, let r = rhr[day], let q = rr[day], let s = slp[day] else { return nil }
            return Night(id: day, hrv: mean(hrv[day]!), rhr: mean(r), rr: mean(q), sleep: s.reduce(0, +))
        }.sorted { $0.id < $1.id }

        // Score each night against the up to 30 nights before it
        for i in asc.indices {
            let prior = asc[max(0, i - 30)..<i]
            guard !prior.isEmpty else { continue }
            let n = Double(prior.count)
            let hb = prior.map(\.hrv).reduce(0, +) / n
            let rb = prior.map(\.rhr).reduce(0, +) / n
            let qb = prior.map(\.rr).reduce(0, +) / n
            let t = asc[i]
            let x = 0.4 * (t.hrv / hb - 1) + 0.25 * (rb / t.rhr - 1) + 0.1 * (qb / t.rr - 1)
            let s = 70 + 150 * x + 60 * (min(t.sleep / 8, 1) - 0.9)
            asc[i].score = Int(max(1, min(100, s)).rounded())
        }
        return asc.reversed()
    }
}

struct ContentView: View {
    @State var nights: [Night] = []
    @State var loading = true
    let health = Health()

    func row(_ a: String, _ b: String, _ c: String, _ d: String, _ e: String, _ f: String) -> some View {
        HStack {
            ForEach([a, b, c, d, e, f], id: \.self) { Text($0).frame(maxWidth: .infinity) }
        }.font(.caption.monospacedDigit())
    }

    var body: some View {
        List {
            row("Date", "Score", "HRV", "RHR", "Resp", "Sleep h").bold()
            ForEach(nights) { n in
                row(n.id.formatted(.dateTime.day().month()), n.score.map(String.init) ?? "-",
                    String(format: "%.0f", n.hrv), String(format: "%.0f", n.rhr),
                    String(format: "%.1f", n.rr), String(format: "%.1f", n.sleep))
            }
        }
        .overlay { if loading { ProgressView() } else if nights.isEmpty { Text("No data") } }
        .task { nights = await health.nights(90); loading = false }
    }
}
