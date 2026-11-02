import SwiftUI
import HealthKit

@main struct RecoveryApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct Night: Identifiable {
    let id: Date
    let hrv, rhr, rr, sleep: Double
    var z: [Double]?          // HRV, sleeping HR, breathing, sleep (higher is better)
    var hp = 0.0, rd = 0.0    // HRV % vs usual, heart rate bpm vs usual
    var score: Int?
    var why = ""
    var color: Color {
        guard let s = score else { return .secondary }
        return s >= 67 ? .green : s >= 34 ? .yellow : .red
    }
}

struct Session: Identifiable {
    let id: Date
    let mins: Double
    let avg, peak: Double?
}

// MARK: Scoring helpers

let defaultW = [0.55, 0.2, 0.05, 0.2]

func key(_ d: Date) -> String {
    let c = Calendar.current.dateComponents([.year, .month, .day], from: d)
    return "\(c.year!)-\(c.month!)-\(c.day!)"
}

func corr(_ a: [Double], _ b: [Double]) -> Double {
    let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
    let cov = zip(a, b).map { ($0 - ma) * ($1 - mb) }.reduce(0, +)
    let va = a.map { ($0 - ma) * ($0 - ma) }.reduce(0, +)
    let vb = b.map { ($0 - mb) * ($0 - mb) }.reduce(0, +)
    return va * vb > 0 ? cov / sqrt(va * vb) : 0
}

// Default weights, blended 50/50 with what predicts your own ratings once you have 14
func weights(_ ns: [Night], _ feel: [String: Int]) -> [Double] {
    let r = ns.compactMap { n -> ([Double], Double)? in
        guard let z = n.z, let f = feel[key(n.id)] else { return nil }
        return (z, Double(f))
    }
    guard r.count >= 14 else { return defaultW }
    let ys = r.map { $0.1 }
    let pos = (0..<4).map { j in max(0, corr(r.map { $0.0[j] }, ys)) }
    let s = pos.reduce(0, +)
    guard s > 0 else { return defaultW }
    return (0..<4).map { 0.5 * defaultW[$0] + 0.5 * pos[$0] / s }
}

func score(_ z: [Double], _ w: [Double]) -> Int {
    let x = 1.5 * zip(z, w).map { $0 * $1 }.reduce(0, +) + 0.2
    let p = 50 * (1 + erf(x / sqrt(2.0)))
    return Int(max(1, min(99, p)).rounded())
}

func reason(_ t: Night, _ z: [Double], _ w: [Double]) -> String {
    var rs: [(Double, String)] = []
    let hpWord = t.hp >= 0 ? "above" : "below"
    let rdWord = t.rd <= 0 ? "lower" : "higher"
    let hours = String(format: "%.1f", t.sleep)
    if abs(z[0]) >= 0.5 { rs.append((abs(w[0] * z[0]), "HRV was \(Int(abs(t.hp).rounded()))% \(hpWord) your usual")) }
    if abs(z[1]) >= 0.5 { rs.append((abs(w[1] * z[1]), "Sleeping heart rate was \(Int(abs(t.rd).rounded())) bpm \(rdWord) than usual")) }
    if z[2] <= -0.5 { rs.append((abs(w[2] * z[2]), "Breathing was faster than usual")) }
    if t.sleep >= 8 { rs.append((abs(w[3] * z[3]), "You slept a full \(hours) h")) }
    if t.sleep < 7 { rs.append((abs(w[3] * z[3]), "Short sleep: \(hours) h against an 8 h target")) }
    if rs.isEmpty { return "Everything was close to your usual." }
    return rs.sorted { $0.0 > $1.0 }.prefix(3).map { $0.1 }.joined(separator: ".\n") + "."
}

// MARK: HealthKit

struct Health {
    let store = HKHealthStore()
    let cal = Calendar.current

    func quantity(_ id: HKQuantityTypeIdentifier, _ unit: HKUnit, _ from: Date) async -> [(Date, Double)] {
        let p = HKQuery.predicateForSamples(withStart: from, end: nil)
        let d = HKSampleQueryDescriptor(predicates: [.quantitySample(type: HKQuantityType(id), predicate: p)], sortDescriptors: [], limit: HKObjectQueryNoLimit)
        return ((try? await d.result(for: store)) ?? []).map { ($0.startDate, $0.quantity.doubleValue(for: unit)) }
    }

    // Asleep periods, sorted and merged so phone and watch overlaps count once
    func sleep(_ from: Date) async -> [(Date, Date)] {
        let p = HKQuery.predicateForSamples(withStart: from, end: nil)
        let d = HKSampleQueryDescriptor(predicates: [.categorySample(type: HKCategoryType(.sleepAnalysis), predicate: p)], sortDescriptors: [], limit: HKObjectQueryNoLimit)
        let asleep = HKCategoryValueSleepAnalysis.allAsleepValues.map(\.rawValue)
        let s = ((try? await d.result(for: store)) ?? []).filter { asleep.contains($0.value) }
            .map { ($0.startDate, $0.endDate) }.sorted { $0.0 < $1.0 }
        var m: [(Date, Date)] = []
        for i in s {
            if let l = m.last, i.0 <= l.1 { m[m.count - 1].1 = max(l.1, i.1) } else { m.append(i) }
        }
        return m
    }

    // Maps a time to its wake day. Only 8pm to 10am counts, so naps are ignored.
    func wakeDay(_ t: Date) -> Date? {
        let x = t.addingTimeInterval(4 * 3600), d = cal.startOfDay(for: x)
        return x.timeIntervalSince(d) < 14 * 3600 ? d : nil
    }

    func inSleep(_ t: Date, _ iv: [(Date, Date)]) -> Bool {
        var lo = 0, hi = iv.count
        while lo < hi { let m = (lo + hi) / 2; if iv[m].0 <= t { lo = m + 1 } else { hi = m } }
        return lo > 0 && t <= iv[lo - 1].1
    }

    func bucket(_ s: [(Date, Double)], _ iv: [(Date, Date)]) -> [Date: [Double]] {
        var r: [Date: [Double]] = [:]
        for (t, v) in s where inSleep(t, iv) { if let d = wakeDay(t) { r[d, default: []].append(v) } }
        return r
    }

    func mean(_ a: [Double]) -> Double { a.reduce(0, +) / Double(a.count) }

    // Value as standard deviations from baseline, capped at +/-3
    func z(_ x: Double, _ base: [Double], floor: Double) -> Double {
        let m = mean(base)
        let sd = sqrt(base.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(base.count))
        return max(-3, min(3, (x - m) / max(sd, floor)))
    }

    func nights(_ days: Int) async -> [Night] {
        let types: Set<HKObjectType> = [HKQuantityType(.heartRateVariabilitySDNN), HKQuantityType(.heartRate), HKQuantityType(.respiratoryRate), HKCategoryType(.sleepAnalysis), HKObjectType.workoutType()]
        try? await store.requestAuthorization(toShare: [], read: types)
        let from = cal.date(byAdding: .day, value: -(days + 1), to: cal.startOfDay(for: Date()))!
        let bpm = HKUnit.count().unitDivided(by: .minute())

        async let h = quantity(.heartRateVariabilitySDNN, .secondUnit(with: .milli), from)
        async let r = quantity(.heartRate, bpm, from)
        async let q = quantity(.respiratoryRate, bpm, from)
        async let sl = sleep(from)
        let iv = await sl
        let hrv = bucket(await h, iv), hr = bucket(await r, iv), rr = bucket(await q, iv)

        var slp: [Date: Double] = [:]
        for (a, b) in iv { if let d = wakeDay(a) { slp[d, default: 0] += b.timeIntervalSince(a) / 3600 } }

        var asc: [Night] = slp.keys.compactMap { day in
            guard day >= from, let h = hrv[day], let r = hr[day], r.count >= 3, let q = rr[day] else { return nil }
            let low = Array(r.sorted().prefix(max(1, r.count / 10)))
            return Night(id: day, hrv: mean(h), rhr: mean(low), rr: mean(q), sleep: slp[day]!)
        }.sorted { $0.id < $1.id }

        // Compare each night to up to 30 earlier nights, after 4 calibration nights
        for i in asc.indices {
            let prior = Array(asc[max(0, i - 30)..<i])
            guard prior.count >= 4 else { continue }
            let t = asc[i]
            let zh = z(log(t.hrv), prior.map { log($0.hrv) }, floor: 0.05)
            let zr = -z(t.rhr, prior.map(\.rhr), floor: 1.5)
            let zq = min(0, -z(t.rr, prior.map(\.rr), floor: 0.4))
            let zs = max(-3, (min(t.sleep / 8, 1) - 0.85) / 0.15)
            asc[i].z = [zh, zr, zq, zs]
            asc[i].hp = (t.hrv / mean(prior.map(\.hrv)) - 1) * 100
            asc[i].rd = t.rhr - mean(prior.map(\.rhr))
        }
        return asc.reversed()
    }

    // Strength workouts logged by your Apple Watch, newest first
    func sessions(_ days: Int) async -> [Session] {
        let from = cal.date(byAdding: .day, value: -days, to: cal.startOfDay(for: Date()))!
        let kind = NSCompoundPredicate(orPredicateWithSubpredicates: [
            HKQuery.predicateForWorkouts(with: .traditionalStrengthTraining),
            HKQuery.predicateForWorkouts(with: .functionalStrengthTraining)])
        let p = NSCompoundPredicate(andPredicateWithSubpredicates: [HKQuery.predicateForSamples(withStart: from, end: nil), kind])
        let d = HKSampleQueryDescriptor(predicates: [.workout(p)], sortDescriptors: [SortDescriptor(\HKWorkout.startDate, order: .reverse)], limit: HKObjectQueryNoLimit)
        let bpm = HKUnit.count().unitDivided(by: .minute())
        return ((try? await d.result(for: store)) ?? []).map { w in
            let s = w.statistics(for: HKQuantityType(.heartRate))
            return Session(id: w.startDate, mins: w.duration / 60,
                           avg: s?.averageQuantity()?.doubleValue(for: bpm),
                           peak: s?.maximumQuantity()?.doubleValue(for: bpm))
        }
    }
}

// MARK: Views

struct NightDetail: View {
    let n: Night
    @Binding var rating: Int
    var body: some View {
        List {
            Text(n.score.map(String.init) ?? "-").font(.system(size: 64, weight: .bold)).foregroundStyle(n.color)
            Text(n.score == nil ? "Needs 4 earlier nights to calibrate." : n.why).foregroundStyle(.secondary)
            Picker("Felt on waking", selection: $rating) {
                Text("-").tag(0)
                ForEach(1...5, id: \.self) { Text("\($0)").tag($0) }
            }.pickerStyle(.segmented)
            LabeledContent("HRV", value: String(format: "%.0f ms", n.hrv))
            LabeledContent("Sleeping heart rate", value: String(format: "%.0f bpm", n.rhr))
            LabeledContent("Respiratory rate", value: String(format: "%.1f /min", n.rr))
            LabeledContent("Sleep", value: String(format: "%.1f h", n.sleep))
        }.navigationTitle(n.id.formatted(.dateTime.day().month()))
    }
}

struct ContentView: View {
    @State var nights: [Night] = []
    @State var sessions: [Session] = []
    @State var loading = true
    @State var feel: [String: Int] = UserDefaults.standard.object(forKey: "feel") as? [String: Int] ?? [:]
    let health = Health()

    // Scores are computed here so a new rating updates them instantly
    var scored: [Night] {
        let w = weights(nights, feel)
        return nights.map { orig in
            var n = orig
            if let z = n.z { n.score = score(z, w); n.why = reason(n, z, w) }
            return n
        }
    }

    func rating(_ d: Date) -> Binding<Int> {
        Binding(get: { feel[key(d)] ?? 0 },
                set: { feel[key(d)] = $0 == 0 ? nil : $0; UserDefaults.standard.set(feel, forKey: "feel") })
    }

    // Weeks start on Monday, newest week first
    var weeks: [(start: Date, nights: [Night])] {
        var c = Calendar(identifier: .gregorian)
        c.firstWeekday = 2
        let g = Dictionary(grouping: scored) { c.dateInterval(of: .weekOfYear, for: $0.id)!.start }
        return g.keys.sorted(by: >).map { ($0, g[$0]!) }
    }

    func summary(_ w: [Night]) -> String {
        guard w.count >= 3 else { return "not enough data" }
        let h = w.map(\.sleep).reduce(0, +) / Double(w.count)
        let r = h >= 8 ? "Very good" : h >= 7.5 ? "Good" : h >= 7 ? "OK" : h >= 6.5 ? "Bad" : "Horrible"
        return "\(r) (avg \(String(format: "%.1f", h)) h)"
    }

    func hm(_ h: Double) -> String {
        let m = Int((h * 60).rounded())
        return String(format: "%d:%02d", m / 60, m % 60)
    }

    func bpm(_ x: Double?) -> String { x.map { String(Int($0)) } ?? "-" }

    // Drift: last 7 nights of HRV vs the 60 nights before, with a 0.5 SD threshold
    var drift: String? {
        let ln = nights.map { log($0.hrv) }
        guard ln.count >= 21 else { return nil }
        let base = Array(ln.dropFirst(7).prefix(60))
        let m = base.reduce(0, +) / Double(base.count)
        let sd = sqrt(base.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(base.count))
        let recent = ln.prefix(7).reduce(0, +) / 7
        return recent < m - 0.5 * max(sd, 0.05) ? "HRV has run below your usual this past week. Watch sleep and training load." : nil
    }

    var learnText: String {
        let n = nights.filter { $0.z != nil && feel[key($0.id)] != nil }.count
        return n >= 14 ? "Score weights tuned from \(n) ratings." : "Rate \(14 - n) more mornings to personalize the score."
    }

    // Learned sleep need: average sleep on your best nights, kept between 7 and 9.5 h
    var learned: Bool { scored.filter { $0.score != nil }.count >= 14 }

    var baseNeed: Double {
        let r = Array(scored.filter { $0.score != nil }.prefix(60))
        guard r.count >= 14 else { return 8 }
        var good = r.filter { (feel[key($0.id)] ?? 0) >= 4 }
        if good.count < 10 { good = Array(r.sorted { $0.score! > $1.score! }.prefix(r.count / 3)) }
        return min(9.5, max(7, good.map(\.sleep).reduce(0, +) / Double(good.count)))
    }

    // Sleep need = baseline + sleep debt + training, like Whoop's structure (no naps yet)
    var need: (base: Double, total: Double, debt: Double, train: Double) {
        let b = baseNeed
        let week = nights.filter { $0.id > Date().addingTimeInterval(-7 * 86400) }
        let debt = min(1.5, 0.25 * week.map { max(0, b - $0.sleep) }.reduce(0, +))
        let mins = sessions.filter { Calendar.current.isDateInToday($0.id) }.map(\.mins).reduce(0, +)
        let train = min(0.75, mins / 120)
        return (b, b + debt + train, debt, train)
    }

    var sleepTab: some View {
        let ws = weeks
        return List {
            if let last = nights.first {
                Section("How did you feel on \(last.id.formatted(.dateTime.weekday().day().month()))? (1 bad, 5 great)") {
                    Picker("Feel", selection: rating(last.id)) {
                        Text("-").tag(0)
                        ForEach(1...5, id: \.self) { Text("\($0)").tag($0) }
                    }.pickerStyle(.segmented)
                    if let d = drift { Text(d) }
                    Text(learnText).font(.caption).foregroundStyle(.secondary)
                }
            }
            ForEach(ws.indices, id: \.self) { i in
                Section("Week of \(ws[i].start.formatted(.dateTime.day().month())): \(summary(ws[i].nights))") {
                    ForEach(ws[i].nights) { n in
                        NavigationLink {
                            NightDetail(n: n, rating: rating(n.id))
                        } label: {
                            HStack {
                                Text(n.id.formatted(.dateTime.weekday().day().month()))
                                Spacer()
                                Text(String(format: "%.1f h", n.sleep)).foregroundStyle(.secondary)
                                Text(n.score.map(String.init) ?? "-").bold().foregroundStyle(n.color).frame(width: 44, alignment: .trailing)
                            }
                        }
                    }
                }
            }
        }.overlay { if loading { ProgressView() } else if nights.isEmpty { Text("No data") } }
    }

    var gymTab: some View {
        List(sessions) { s in
            VStack(alignment: .leading) {
                Text(s.id.formatted(.dateTime.weekday().day().month())).bold()
                Text("\(Int(s.mins)) min, avg \(bpm(s.avg)) bpm, peak \(bpm(s.peak)) bpm").font(.caption).foregroundStyle(.secondary)
            }
        }.overlay { if !loading && sessions.isEmpty { Text("No strength workouts") } }
    }

    var needTab: some View {
        let n = need
        return VStack(spacing: 8) {
            Text("Sleep tonight").font(.headline)
            Text(hm(n.total)).font(.system(size: 72, weight: .bold))
            Text("Baseline \(hm(n.base)), debt +\(hm(n.debt)), training +\(hm(n.train))").font(.caption).foregroundStyle(.secondary)
            Text(learned ? "Baseline learned from your best nights." : "Baseline is 8:00 until you have 14 scored nights.").font(.caption).foregroundStyle(.secondary)
        }
    }

    var body: some View {
        TabView {
            NavigationStack { sleepTab.navigationTitle("Sleep") }.tabItem { Label("Sleep", systemImage: "bed.double.fill") }
            NavigationStack { gymTab.navigationTitle("Gym") }.tabItem { Label("Gym", systemImage: "dumbbell.fill") }
            NavigationStack { needTab.navigationTitle("Tonight") }.tabItem { Label("Need", systemImage: "moon.zzz.fill") }
        }
        .task { nights = await health.nights(90); sessions = await health.sessions(90); loading = false }
    }
}
