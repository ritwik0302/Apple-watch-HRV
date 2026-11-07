import SwiftUI
import HealthKit
import Charts

@main
struct RecoveryApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

// MARK: - Data models

struct Night: Identifiable {
    let id: Date
    let hrv: Double
    let rhr: Double
    let respiratoryRate: Double
    let sleepHours: Double
    var features: [Double]?       // HRV, sleeping HR, breathing, sleep (higher is better)
    var hrvChangePercent = 0.0
    var rhrChangeBPM = 0.0
    var score: Int?
    var explanation = ""

    var color: Color {
        guard let score else { return Theme.muted }
        return score >= 67 ? Theme.go : score >= 34 ? Theme.caution : Theme.stop
    }
}

struct WorkoutSession: Identifiable {
    let id: Date
    let minutes: Double
    let averageHeartRate: Double?
    let peakHeartRate: Double?
}

// MARK: - Personal learning

private let defaultWeights = [0.55, 0.20, 0.05, 0.20]

func dayKey(_ date: Date) -> String {
    let c = Calendar.current.dateComponents([.year, .month, .day], from: date)
    return "\(c.year ?? 0)-\(c.month ?? 0)-\(c.day ?? 0)"
}

func mean(_ values: [Double]) -> Double? {
    guard !values.isEmpty else { return nil }
    return values.reduce(0, +) / Double(values.count)
}

func correlation(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, a.count >= 3,
          let ma = mean(a), let mb = mean(b) else { return 0 }
    let covariance = zip(a, b).reduce(0.0) { $0 + ($1.0 - ma) * ($1.1 - mb) }
    let va = a.reduce(0.0) { $0 + pow($1 - ma, 2) }
    let vb = b.reduce(0.0) { $0 + pow($1 - mb, 2) }
    guard va > 0, vb > 0 else { return 0 }
    return covariance / sqrt(va * vb)
}

/// Learns feature weights from this user's own morning ratings.
/// Only the newest 90 ratings are used, and the learned part is capped at 50% so a few
/// odd mornings cannot flip the score.
func personalWeights(nights: [Night], ratings: [String: Int]) -> [Double] {
    let examples: [([Double], Double)] = nights.compactMap { night in
        guard let x = night.features,
              let rating = ratings[dayKey(night.id)], rating > 0 else { return nil }
        return (x, Double(rating))
    }
    guard examples.count >= 8 else { return defaultWeights }

    let recent = Array(examples.prefix(90))   // nights are newest first
    let target = recent.map { $0.1 }
    let relevance = (0..<4).map { index -> Double in
        let values = recent.map { $0.0[index] }
        return max(0, correlation(values, target))
    }
    let total = relevance.reduce(0, +)
    guard total > 0 else { return defaultWeights }
    let learned = relevance.map { $0 / total }

    let learningStrength = min(0.5, Double(recent.count) / 60.0)
    let blended = (0..<4).map { index in
        (1 - learningStrength) * defaultWeights[index] + learningStrength * learned[index]
    }
    let normalizer = blended.reduce(0, +)
    return blended.map { $0 / normalizer }
}

/// Turns the user's own data and ratings into an explainable learning status.
func personalInsight(nights: [Night], ratings: [String: Int]) -> String {
    let eligible = nights.filter { $0.features != nil }
    let rated = eligible.compactMap { night -> (Night, Int)? in
        guard let rating = ratings[dayKey(night.id)], rating > 0 else { return nil }
        return (night, rating)
    }
    guard rated.count >= 5 else {
        return "I'm building your personal baseline. Rate how you feel each morning so I can learn which signals matter most to you."
    }
    let recent = Array(rated.prefix(7))   // newest first
    let averageRating = mean(recent.map { Double($0.1) }) ?? 0
    let hrvRatings = rated.filter { $0.0.hrvChangePercent.isFinite }
    let hrvCorrelation = correlation(hrvRatings.map { $0.0.hrvChangePercent }, hrvRatings.map { Double($0.1) })
    let sleepCorrelation = correlation(rated.map { $0.0.sleepHours }, rated.map { Double($0.1) })

    var observations: [String] = []
    if rated.count >= 8 && abs(hrvCorrelation) >= 0.25 {
        observations.append(hrvCorrelation > 0
            ? "Your ratings have tended to be better when HRV is higher than your baseline"
            : "Your ratings haven't followed the usual HRV pattern consistently")
    }
    if rated.count >= 8 && sleepCorrelation >= 0.25 {
        observations.append("More sleep has tended to line up with better morning ratings")
    }
    if observations.isEmpty {
        return "I've learned from \(rated.count) rated mornings. Your personal patterns are still emerging, so I'm keeping the predictions conservative."
    }
    return "From your \(rated.count) rated mornings: " + observations.prefix(2).joined(separator: ". ") + ". Your last \(recent.count) ratings average \(String(format: "%.1f", averageRating))/5."
}

// MARK: - Sleep need (statistical model)

struct SleepPrediction {
    let hours: Double
    let baseline: Double
    let debt: Double
    let training: Double
    let samples: Int
    let isPersonalised: Bool
}

private struct SleepTrainingExample {
    let sleepHours: Double
    let workoutHours: Double
    let sleepDebt: Double
    let rating: Double
}

/// Fits a small regularised regression to the user's own rated mornings.
private func fitSleepModel(_ examples: [SleepTrainingExample]) -> [Double]? {
    let size = 5
    var matrix = Array(repeating: Array(repeating: 0.0, count: size), count: size)
    var vector = Array(repeating: 0.0, count: size)

    for example in examples {
        let sleep = example.sleepHours - 8.0
        let x = [1.0, sleep, sleep * sleep, example.workoutHours, example.sleepDebt]
        for row in 0..<size {
            vector[row] += x[row] * example.rating
            for column in 0..<size {
                matrix[row][column] += x[row] * x[column]
            }
        }
    }

    // Ridge regularisation limits unstable coefficients with small datasets.
    for index in 1..<size { matrix[index][index] += 2.0 }

    // Gaussian elimination with partial pivoting.
    for column in 0..<size {
        var pivot = column
        for row in (column + 1)..<size {
            if abs(matrix[row][column]) > abs(matrix[pivot][column]) { pivot = row }
        }
        guard abs(matrix[pivot][column]) > 0.000001 else { return nil }
        if pivot != column {
            matrix.swapAt(pivot, column)
            vector.swapAt(pivot, column)
        }

        let divisor = matrix[column][column]
        for j in column..<size { matrix[column][j] /= divisor }
        vector[column] /= divisor

        for row in 0..<size where row != column {
            let factor = matrix[row][column]
            for j in column..<size { matrix[row][j] -= factor * matrix[column][j] }
            vector[row] -= factor * vector[column]
        }
    }
    return vector.allSatisfy { $0.isFinite } ? vector : nil
}

func personalSleepPrediction(
    nights: [Night],
    workouts: [WorkoutSession],
    ratings: [String: Int],
    baseline: Double,
    today: Date = Date()
) -> SleepPrediction {
    let calendar = Calendar.current
    let todayStart = calendar.startOfDay(for: today)

    func workoutHours(on date: Date) -> Double {
        workouts
            .filter { calendar.isDate($0.id, inSameDayAs: date) }
            .map(\.minutes)
            .reduce(0, +) / 60.0
    }

    func debt(before date: Date) -> Double {
        let start = calendar.date(byAdding: .day, value: -7, to: date) ?? date
        let previous = nights.filter { $0.id >= start && $0.id < date }
        return min(14.0, previous.reduce(0.0) { $0 + max(0, baseline - $1.sleepHours) })
    }

    // Pair each night's sleep with the preceding day's training and the
    // user's rating for the morning after that sleep.
    let examples: [SleepTrainingExample] = nights.compactMap { night in
        guard let rating = ratings[dayKey(night.id)], (1...5).contains(rating),
              night.sleepHours > 0,
              let previousDay = calendar.date(byAdding: .day, value: -1, to: night.id) else {
            return nil
        }
        return SleepTrainingExample(
            sleepHours: night.sleepHours,
            workoutHours: workoutHours(on: previousDay),
            sleepDebt: debt(before: night.id),
            rating: Double(rating)
        )
    }

    // Debt for tonight includes this morning's night, so look up to tomorrow.
    let tonight = calendar.date(byAdding: .day, value: 1, to: todayStart) ?? todayStart
    let currentDebt = debt(before: tonight)
    let currentTraining = workoutHours(on: todayStart)
    let extra = min(1.5, currentDebt * 0.25) + min(0.75, currentTraining / 2.0)
    let fallback = min(10.0, max(7.0, baseline + extra))

    guard examples.count >= 12, let coefficients = fitSleepModel(examples) else {
        return SleepPrediction(hours: fallback, baseline: baseline, debt: currentDebt,
                               training: currentTraining, samples: examples.count,
                               isPersonalised: false)
    }

    // The fitted curve peaks at the sleep length your ratings favour. Training and debt are
    // the same for every candidate length, so they are added on top instead of inside a search.
    let curved = coefficients[2] < 0
    var learnedBase = baseline
    if curved {
        let peak = 8.0 - coefficients[1] / (2 * coefficients[2])
        let trust = min(1.0, Double(examples.count) / 40.0)
        learnedBase = trust * min(10, max(7, peak)) + (1 - trust) * baseline
    }
    return SleepPrediction(hours: min(10, max(7, learnedBase + extra)), baseline: learnedBase,
                           debt: currentDebt, training: currentTraining, samples: examples.count,
                           isPersonalised: curved)
}

func recoveryScore(features: [Double], weights: [Double]) -> Int {
    let weighted = zip(features, weights).reduce(0.0) { $0 + $1.0 * $1.1 }
    let x = 1.5 * weighted + 0.2
    let probability = 50 * (1 + erf(x / sqrt(2)))
    return Int(max(1, min(99, probability)).rounded())
}

func recoveryExplanation(_ night: Night, weights: [Double]) -> String {
    guard let f = night.features else { return "Not enough baseline data yet." }
    var reasons: [(Double, String)] = []
    if abs(f[0]) >= 0.5 {
        let direction = night.hrvChangePercent >= 0 ? "above" : "below"
        reasons.append((abs(weights[0] * f[0]), "HRV was \(Int(abs(night.hrvChangePercent).rounded()))% \(direction) your recent baseline"))
    }
    if abs(f[1]) >= 0.5 {
        let direction = night.rhrChangeBPM <= 0 ? "lower" : "higher"
        reasons.append((abs(weights[1] * f[1]), "Sleeping heart rate was \(Int(abs(night.rhrChangeBPM).rounded())) bpm \(direction) than baseline"))
    }
    if f[2] <= -0.5 {
        reasons.append((abs(weights[2] * f[2]), "Respiratory rate was higher than your recent baseline"))
    }
    if night.sleepHours >= 8 {
        reasons.append((abs(weights[3] * f[3]), "You slept \(String(format: "%.1f", night.sleepHours)) hours"))
    } else if night.sleepHours < 7 {
        reasons.append((abs(weights[3] * f[3]), "Short sleep: \(String(format: "%.1f", night.sleepHours)) hours"))
    }
    guard !reasons.isEmpty else { return "Your measurements were close to your recent baseline." }
    return reasons.sorted { $0.0 > $1.0 }.prefix(3).map { $0.1 }.joined(separator: ". ") + "."
}

// MARK: - HealthKit

struct HealthKitManager {
    private let store = HKHealthStore()
    private let calendar = Calendar.current

    func requestAccess() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthError.unavailable
        }
        let readTypes: Set<HKObjectType> = [
            HKQuantityType(.heartRateVariabilitySDNN),
            HKQuantityType(.heartRate),
            HKQuantityType(.respiratoryRate),
            HKCategoryType(.sleepAnalysis),
            HKObjectType.workoutType()
        ]
        try await store.requestAuthorization(toShare: [], read: readTypes)
    }

    func quantitySamples(_ identifier: HKQuantityTypeIdentifier, unit: HKUnit, from start: Date) async -> [(Date, Double)] {
        let predicate = HKQuery.predicateForSamples(withStart: start, end: nil)
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.quantitySample(type: HKQuantityType(identifier), predicate: predicate)],
            sortDescriptors: [SortDescriptor(\HKQuantitySample.startDate)],
            limit: HKObjectQueryNoLimit
        )
        do {
            let samples = try await descriptor.result(for: store)
            return samples.map { ($0.startDate, $0.quantity.doubleValue(for: unit)) }
        } catch {
            print("HealthKit quantity query failed (\(identifier)): \(error)")
            return []
        }
    }

    /// Returns merged asleep intervals so overlapping iPhone and Watch records aren't double-counted.
    func sleepIntervals(from start: Date) async -> [(Date, Date)] {
        let predicate = HKQuery.predicateForSamples(withStart: start, end: nil)
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.categorySample(type: HKCategoryType(.sleepAnalysis), predicate: predicate)],
            sortDescriptors: [SortDescriptor(\HKCategorySample.startDate)],
            limit: HKObjectQueryNoLimit
        )
        do {
            let samples = try await descriptor.result(for: store)
            let asleepValues = HKCategoryValueSleepAnalysis.allAsleepValues.map(\.rawValue)
            let intervals = samples
                .filter { asleepValues.contains($0.value) }
                .map { ($0.startDate, $0.endDate) }
                .sorted { $0.0 < $1.0 }

            var merged: [(Date, Date)] = []
            for interval in intervals {
                if let last = merged.last, interval.0 <= last.1 {
                    merged[merged.count - 1].1 = max(last.1, interval.1)
                } else {
                    merged.append(interval)
                }
            }
            return merged
        } catch {
            print("HealthKit sleep query failed: \(error)")
            return []
        }
    }

    /// Assigns overnight measurements to the morning they belong to. Daytime naps are excluded.
    func wakeDay(for date: Date) -> Date? {
        let adjusted = date.addingTimeInterval(4 * 60 * 60)
        let day = calendar.startOfDay(for: adjusted)
        return adjusted.timeIntervalSince(day) < 14 * 60 * 60 ? day : nil
    }

    func isInsideSleep(_ date: Date, intervals: [(Date, Date)]) -> Bool {
        var low = 0
        var high = intervals.count
        while low < high {
            let middle = (low + high) / 2
            if intervals[middle].0 <= date { low = middle + 1 }
            else { high = middle }
        }
        return low > 0 && date <= intervals[low - 1].1
    }

    func bucket(_ samples: [(Date, Double)], intervals: [(Date, Date)]) -> [Date: [Double]] {
        var result: [Date: [Double]] = [:]
        for (date, value) in samples where isInsideSleep(date, intervals: intervals) {
            if let day = wakeDay(for: date) { result[day, default: []].append(value) }
        }
        return result
    }

    /// Standard-deviation distance from the previous nights, capped so outliers can't dominate.
    func zScore(_ value: Double, baseline: [Double], floor: Double) -> Double? {
        guard let average = mean(baseline), baseline.count >= 4 else { return nil }
        let variance = baseline.reduce(0.0) { $0 + pow($1 - average, 2) } / Double(baseline.count)
        let standardDeviation = sqrt(variance)
        return max(-3, min(3, (value - average) / max(standardDeviation, floor)))
    }

    func loadNights(days: Int) async -> [Night] {
        do { try await requestAccess() }
        catch {
            print("HealthKit authorization unavailable: \(error)")
            return []
        }

        let start = calendar.date(byAdding: .day, value: -(days + 1), to: calendar.startOfDay(for: Date())) ?? Date()
        let bpm = HKUnit.count().unitDivided(by: .minute())

        async let hrvSamples = quantitySamples(.heartRateVariabilitySDNN, unit: .secondUnit(with: .milli), from: start)
        async let hrSamples = quantitySamples(.heartRate, unit: bpm, from: start)
        async let respiratorySamples = quantitySamples(.respiratoryRate, unit: bpm, from: start)
        async let intervalsTask = sleepIntervals(from: start)

        let intervals = await intervalsTask
        let hrvByDay = bucket(await hrvSamples, intervals: intervals)
        let hrByDay = bucket(await hrSamples, intervals: intervals)
        let respiratoryByDay = bucket(await respiratorySamples, intervals: intervals)

        var sleepByDay: [Date: Double] = [:]
        for (begin, end) in intervals {
            if let day = wakeDay(for: begin) {
                sleepByDay[day, default: 0] += end.timeIntervalSince(begin) / 3600
            }
        }

        var nights: [Night] = sleepByDay.keys.compactMap { day in
            guard day >= start,
                  let hrvValues = hrvByDay[day], !hrvValues.isEmpty,
                  let heartRates = hrByDay[day], heartRates.count >= 3,
                  let respiratoryValues = respiratoryByDay[day], !respiratoryValues.isEmpty,
                  let sleep = sleepByDay[day], sleep > 0 else { return nil }

            // Use the lowest 10% of sleeping HR samples as a stable resting estimate.
            let lowestHeartRates = Array(heartRates.sorted().prefix(max(1, heartRates.count / 10)))
            guard let hrv = mean(hrvValues), let rhr = mean(lowestHeartRates),
                  let respiratory = mean(respiratoryValues) else { return nil }
            return Night(id: day, hrv: hrv, rhr: rhr, respiratoryRate: respiratory, sleepHours: sleep)
        }.sorted { $0.id < $1.id }

        for index in nights.indices {
            let previous = Array(nights[max(0, index - 30)..<index])
            guard previous.count >= 4 else { continue }
            let current = nights[index]
            guard current.hrv > 0,
                  let hrvZ = zScore(log(current.hrv), baseline: previous.map { log($0.hrv) }, floor: 0.05),
                  let rhrZ = zScore(current.rhr, baseline: previous.map(\.rhr), floor: 1.5),
                  let respiratoryZ = zScore(current.respiratoryRate, baseline: previous.map(\.respiratoryRate), floor: 0.4),
                  let baselineHRV = mean(previous.map(\.hrv)),
                  let baselineRHR = mean(previous.map(\.rhr)) else { continue }

            let sleepFeature = max(-3, (min(current.sleepHours / 8, 1) - 0.85) / 0.15)
            let features = [hrvZ, -rhrZ, min(0, -respiratoryZ), sleepFeature]
            nights[index].features = features
            nights[index].hrvChangePercent = (current.hrv / baselineHRV - 1) * 100
            nights[index].rhrChangeBPM = current.rhr - baselineRHR
        }
        return nights.reversed()
    }

    func loadStrengthWorkouts(days: Int) async -> [WorkoutSession] {
        let start = calendar.date(byAdding: .day, value: -days, to: calendar.startOfDay(for: Date())) ?? Date()
        let workoutTypes = NSCompoundPredicate(orPredicateWithSubpredicates: [
            HKQuery.predicateForWorkouts(with: .traditionalStrengthTraining),
            HKQuery.predicateForWorkouts(with: .functionalStrengthTraining)
        ])
        let datePredicate = HKQuery.predicateForSamples(withStart: start, end: nil)
        let predicate = NSCompoundPredicate(andPredicateWithSubpredicates: [datePredicate, workoutTypes])
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.workout(predicate)],
            sortDescriptors: [SortDescriptor(\HKWorkout.startDate, order: .reverse)],
            limit: HKObjectQueryNoLimit
        )
        let bpm = HKUnit.count().unitDivided(by: .minute())
        do {
            return try await descriptor.result(for: store).map { workout in
                let statistics = workout.statistics(for: HKQuantityType(.heartRate))
                return WorkoutSession(
                    id: workout.startDate,
                    minutes: workout.duration / 60,
                    averageHeartRate: statistics?.averageQuantity()?.doubleValue(for: bpm),
                    peakHeartRate: statistics?.maximumQuantity()?.doubleValue(for: bpm)
                )
            }
        } catch {
            print("HealthKit workout query failed: \(error)")
            return []
        }
    }
}

enum HealthError: Error {
    case unavailable
}

// MARK: - Design

enum Theme {
    static let bg = Color(hex: 0x0F1730)
    static let card = Color(hex: 0x182244)
    static let text = Color(hex: 0xEDEBE4)
    static let muted = Color(hex: 0x8F98B5)
    static let go = Color(hex: 0x58D6A6)
    static let caution = Color(hex: 0xF3B94D)
    static let stop = Color(hex: 0xF26D5B)
    static let ai = Color(hex: 0xA797FF)

    static func serif(_ size: CGFloat, _ weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255)
    }
}

let moodFaces = ["😫", "😕", "😐", "🙂", "🤩"]

func bandName(_ score: Int?) -> String {
    guard let s = score else { return "Calibrating" }
    return s >= 67 ? "Ready" : s >= 34 ? "Steady" : "Rest"
}

struct Card<Content: View>: View {
    private let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

/// The one memorable element: a cockpit-style dial whose ticks fill up to the score.
struct ScoreDial: View {
    let score: Int?
    let color: Color
    var size: CGFloat = 250
    @State private var shown = 0.0

    private var target: Double { Double(score ?? 0) / 100 }

    var body: some View {
        ZStack {
            Circle()
                .trim(from: 0, to: 0.75)
                .stroke(Theme.muted.opacity(0.25), style: StrokeStyle(lineWidth: size * 0.08, dash: [2, 5]))
                .rotationEffect(.degrees(135))
                .padding(size * 0.05)
            Circle()
                .trim(from: 0, to: 0.75 * shown)
                .stroke(color, style: StrokeStyle(lineWidth: size * 0.08, dash: [2, 5]))
                .rotationEffect(.degrees(135))
                .padding(size * 0.05)
            VStack(spacing: 2) {
                Text(score.map(String.init) ?? "-")
                    .font(Theme.serif(size * 0.34, .bold))
                    .foregroundStyle(Theme.text)
                    .contentTransition(.numericText())
                Text(bandName(score))
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(color)
            }
        }
        .frame(width: size, height: size)
        .onAppear { withAnimation(.easeOut(duration: 1.3)) { shown = target } }
        .onChange(of: score) { _, _ in withAnimation(.easeOut(duration: 0.8)) { shown = target } }
    }
}

struct MoodPicker: View {
    @Binding var rating: Int

    var body: some View {
        HStack(spacing: 8) {
            ForEach(1...5, id: \.self) { value in
                Button {
                    rating = rating == value ? 0 : value
                } label: {
                    Text(moodFaces[value - 1])
                        .font(.system(size: 30))
                        .frame(maxWidth: .infinity, minHeight: 54)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(rating == value ? Theme.card : Color.clear)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .stroke(rating == value ? Theme.text.opacity(0.6) : Theme.muted.opacity(0.25), lineWidth: 1)
                        )
                        .scaleEffect(rating == value ? 1.12 : 1)
                        .opacity(rating == 0 || rating == value ? 1 : 0.45)
                }
                .buttonStyle(.plain)
            }
        }
        .animation(.spring(duration: 0.3), value: rating)
        .sensoryFeedback(.selection, trigger: rating)
    }
}

struct Perk: Identifiable {
    let id: String
    let icon: String
    let title: String
    let detail: String
    let earned: Bool
}

// MARK: - Views

struct NightDetail: View {
    let night: Night
    @Binding var rating: Int
    let predicted: Double?

    private func stat(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).foregroundStyle(Theme.muted)
            Spacer()
            Text(value).font(Theme.serif(17)).monospacedDigit()
        }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                ScoreDial(score: night.score, color: night.color, size: 220)
                Text(night.score == nil ? "Needs four earlier nights to find your baseline." : night.explanation)
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("How did you feel on waking?").font(.subheadline).foregroundStyle(Theme.muted)
                        MoodPicker(rating: $rating)
                    }
                }
                Card {
                    VStack(spacing: 12) {
                        stat("HRV", String(format: "%.0f ms", night.hrv))
                        stat("Sleeping heart rate", String(format: "%.0f bpm", night.rhr))
                        stat("Respiratory rate", String(format: "%.1f /min", night.respiratoryRate))
                        stat("Sleep", String(format: "%.1f h", night.sleepHours))
                        Divider()
                        HStack {
                            Label("AI predicted feeling", systemImage: "sparkles").foregroundStyle(Theme.ai)
                            Spacer()
                            Text(predicted.map { String(format: "%.1f / 5", $0) } ?? "-")
                                .font(Theme.serif(17)).monospacedDigit()
                        }
                    }
                }
            }
            .padding()
        }
        .background(Theme.bg)
        .navigationTitle(night.id.formatted(.dateTime.day().month()))
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ContentView: View {
    @State private var nights: [Night] = []
    @State private var workouts: [WorkoutSession] = []
    @State private var isLoading = true
    @State private var loadError: String?
    @State private var ratings: [String: Int] = UserDefaults.standard.object(forKey: "recoveryRatings") as? [String: Int] ?? [:]
    @State private var openPerk: String?
    @AppStorage("wakeMinutes") private var wakeMinutes = 7 * 60
    @StateObject private var ai = PersonalModel()

    private let health = HealthKitManager()

    // Scores are computed here, so a new rating updates them instantly.
    private var scoredNights: [Night] {
        let w = personalWeights(nights: nights, ratings: ratings)
        return nights.map { original in
            var night = original
            if let features = night.features {
                night.score = recoveryScore(features: features, weights: w)
                night.explanation = recoveryExplanation(night, weights: w)
            }
            return night
        }
    }

    private func ratingBinding(for date: Date) -> Binding<Int> {
        Binding(
            get: { ratings[dayKey(date)] ?? 0 },
            set: { newValue in
                if newValue == 0 { ratings.removeValue(forKey: dayKey(date)) }
                else { ratings[dayKey(date)] = newValue }
                UserDefaults.standard.set(ratings, forKey: "recoveryRatings")
            }
        )
    }

    // Weeks start on Monday, newest week first
    private var weeks: [(start: Date, nights: [Night])] {
        var calendar = Calendar(identifier: .gregorian)
        calendar.firstWeekday = 2
        let grouped = Dictionary(grouping: scoredNights) {
            calendar.dateInterval(of: .weekOfYear, for: $0.id)?.start ?? calendar.startOfDay(for: $0.id)
        }
        return grouped.keys.sorted(by: >).map { ($0, grouped[$0] ?? []) }
    }

    private func weekSummary(_ values: [Night]) -> String {
        guard let average = mean(values.map(\.sleepHours)), values.count >= 3 else { return "Not enough data" }
        let quality = average >= 8 ? "Very good" : average >= 7.5 ? "Good" : average >= 7 ? "OK" : average >= 6.5 ? "Bad" : "Horrible"
        return "\(quality), avg \(String(format: "%.1f", average)) h"
    }

    private func hoursAndMinutes(_ hours: Double) -> String {
        let minutes = Int((hours * 60).rounded())
        return String(format: "%d:%02d", minutes / 60, minutes % 60)
    }

    private func bpmText(_ value: Double?) -> String {
        value.map { String(Int($0.rounded())) } ?? "-"
    }

    // Drift: last 7 nights of HRV against the 60 nights before, with a 0.5 SD threshold
    private var hrvTrendWarning: String? {
        let values = nights.map { log($0.hrv) }
        guard values.count >= 21 else { return nil }
        let recent = Array(values.prefix(7))
        let baseline = Array(values.dropFirst(7).prefix(60))
        guard let recentMean = mean(recent), let baselineMean = mean(baseline) else { return nil }
        let variance = baseline.reduce(0.0) { $0 + pow($1 - baselineMean, 2) } / Double(baseline.count)
        let sd = sqrt(variance)
        return recentMean < baselineMean - 0.5 * max(sd, 0.05)
            ? "Your HRV has been below its recent baseline this week. Consider sleep, stress and training load together."
            : nil
    }

    private var ratingCount: Int {
        nights.filter { $0.features != nil && (ratings[dayKey($0.id)] ?? 0) > 0 }.count
    }

    private var learningDescription: String {
        ratingCount >= 8
            ? "Your recovery weighting is adapting to your \(ratingCount) rated mornings."
            : "Rate \(max(0, 8 - ratingCount)) more eligible mornings to start personal adaptation."
    }

    private var estimatedBaseSleep: Double {
        let eligible = Array(scoredNights.filter { $0.score != nil }.prefix(60))
        guard eligible.count >= 14 else { return 8 }
        var goodNights = eligible.filter { (ratings[dayKey($0.id)] ?? 0) >= 4 }
        if goodNights.count < 10 {
            goodNights = Array(eligible.sorted { ($0.score ?? 0) > ($1.score ?? 0) }.prefix(max(1, eligible.count / 3)))
        }
        return min(9.5, max(7, mean(goodNights.map(\.sleepHours)) ?? 8))
    }

    private var sleepTarget: SleepPrediction {
        personalSleepPrediction(
            nights: nights,
            workouts: workouts,
            ratings: ratings,
            baseline: estimatedBaseSleep
        )
    }

    // On-device neural network: training examples, per-night prediction and sleep suggestion
    private var aiExamples: [([Double], Double)] {
        let base = estimatedBaseSleep
        return nights.compactMap { night in
            guard let rating = ratings[dayKey(night.id)], (1...5).contains(rating),
                  let x = modelInput(night, nights: nights, workouts: workouts, baseline: base) else { return nil }
            return (x, Double(rating))
        }
    }

    private func aiPrediction(_ night: Night) -> Double? {
        modelInput(night, nights: nights, workouts: workouts, baseline: estimatedBaseSleep).flatMap { ai.predict($0) }
    }

    private var aiSleep: Double? {
        guard ai.trainedOn >= 10 else { return nil }
        let target = sleepTarget
        return ai.bestSleep(training: target.training, debt: target.debt)
    }

    // MARK: Style helpers

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        return hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening"
    }

    private func weekColor(_ values: [Night]) -> Color {
        guard let average = mean(values.map(\.sleepHours)), values.count >= 3 else { return Theme.muted }
        return average >= 7.5 ? Theme.go : average >= 6.5 ? Theme.caution : Theme.stop
    }

    // Consecutive recorded nights with 7 hours or more, newest first
    private var sleepStreak: Int {
        var count = 0
        for night in nights {
            if night.sleepHours >= 7 { count += 1 } else { break }
        }
        return count
    }

    private var perks: [Perk] {
        let scores = scoredNights.compactMap { $0.score }
        let weekAgo = Date().addingTimeInterval(-7 * 86400)
        return [
            Perk(id: "green", icon: "bolt.fill", title: "First green day", detail: "Reach a recovery score of 67", earned: scores.contains { $0 >= 67 }),
            Perk(id: "streak", icon: "moon.stars.fill", title: "Seven nights", detail: "Sleep 7 h or more, 7 nights running", earned: sleepStreak >= 7),
            Perk(id: "rated", icon: "face.smiling.fill", title: "Ten mornings", detail: "Rate 10 mornings", earned: ratingCount >= 10),
            Perk(id: "ai", icon: "sparkles", title: "Model awake", detail: "Let the on-device model train on 10 ratings", earned: ai.trainedOn >= 10),
            Perk(id: "gym", icon: "dumbbell.fill", title: "Three sessions", detail: "Train 3 times in the last 7 days", earned: workouts.filter { $0.id > weekAgo }.count >= 3)
        ]
    }

    private func perkButton(_ perk: Perk) -> some View {
        Button {
            openPerk = openPerk == perk.id ? nil : perk.id
        } label: {
            Image(systemName: perk.earned ? perk.icon : "lock.fill")
                .font(.system(size: 18))
                .foregroundStyle(perk.earned ? Theme.bg : Theme.muted)
                .frame(width: 46, height: 46)
                .background(Circle().fill(perk.earned ? Theme.text : Theme.bg))
        }
        .buttonStyle(.plain)
    }

    private func nightRow(_ night: Night) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(night.id.formatted(.dateTime.weekday(.wide)))
                Text(night.id.formatted(.dateTime.day().month())).font(.caption).foregroundStyle(Theme.muted)
            }
            Spacer()
            if let mood = ratings[dayKey(night.id)], mood > 0 {
                Text(moodFaces[mood - 1])
            }
            Text(String(format: "%.1f h", night.sleepHours)).font(.subheadline).foregroundStyle(Theme.muted)
            Text(night.score.map(String.init) ?? "-")
                .font(Theme.serif(16))
                .foregroundStyle(night.color)
                .frame(width: 40, height: 40)
                .background(Circle().fill(night.color.opacity(0.15)))
                .overlay(Circle().stroke(night.color, lineWidth: 1.5))
        }
    }

    // MARK: Tabs

    private var sleepTab: some View {
        let ws = weeks
        let scored = scoredNights
        let last7 = Array(scored.prefix(7).reversed())
        return List {
            if let latest = scored.first {
                VStack(spacing: 14) {
                    ScoreDial(score: latest.score, color: latest.color)
                    Text(latest.score == nil ? "Needs four earlier nights to find your baseline." : latest.explanation)
                        .font(.callout)
                        .foregroundStyle(Theme.muted)
                        .multilineTextAlignment(.center)
                    Text(latest.id.formatted(.dateTime.weekday(.wide).day().month(.wide)))
                        .font(.footnote)
                        .foregroundStyle(Theme.muted)
                }
                .frame(maxWidth: .infinity)
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                VStack(alignment: .leading, spacing: 10) {
                    Text("How did you feel?").font(Theme.serif(20))
                    MoodPicker(rating: ratingBinding(for: latest.id))
                    Text(learningDescription).font(.caption).foregroundStyle(Theme.muted)
                }
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                if let warning = hrvTrendWarning {
                    Card {
                        Label(warning, systemImage: "waveform.path.ecg")
                            .font(.subheadline)
                            .foregroundStyle(Theme.caution)
                    }
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }

                Card {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Last 7 nights").font(Theme.serif(20))
                            Spacer()
                            Text("Dashed line is your need").font(.caption).foregroundStyle(Theme.muted)
                        }
                        Chart {
                            ForEach(last7) { night in
                                BarMark(x: .value("Day", night.id, unit: .day), y: .value("Hours", night.sleepHours))
                                    .foregroundStyle(night.color)
                                    .cornerRadius(4)
                            }
                            RuleMark(y: .value("Need", estimatedBaseSleep))
                                .foregroundStyle(Theme.muted)
                                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                        }
                        .chartYScale(domain: 0...10)
                        .chartYAxis(.hidden)
                        .chartXAxis {
                            AxisMarks(values: .stride(by: .day)) { _ in
                                AxisValueLabel(format: .dateTime.weekday(.narrow))
                            }
                        }
                        .frame(height: 120)
                    }
                }
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label("\(sleepStreak) night streak", systemImage: "moon.stars.fill")
                                .font(Theme.serif(20))
                            Spacer()
                            Text("\(perks.filter { $0.earned }.count) of \(perks.count) badges")
                                .font(.caption)
                                .foregroundStyle(Theme.muted)
                        }
                        HStack(spacing: 12) {
                            ForEach(perks) { perk in perkButton(perk) }
                        }
                        if let open = perks.first(where: { $0.id == openPerk }) {
                            Text("\(open.title): \(open.detail)")
                                .font(.caption)
                                .foregroundStyle(Theme.muted)
                        }
                    }
                }
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
                .sensoryFeedback(.impact(weight: .light), trigger: openPerk)

                Card {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("What your ratings say").font(Theme.serif(20))
                        Text(personalInsight(nights: nights, ratings: ratings)).font(.subheadline)
                        Text("Learns on this iPhone from your Health data and morning feedback. These are wellness patterns, not medical conclusions.")
                            .font(.caption2)
                            .foregroundStyle(Theme.muted)
                    }
                }
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
            }

            ForEach(ws.indices, id: \.self) { index in
                Section {
                    ForEach(ws[index].nights) { night in
                        NavigationLink {
                            NightDetail(night: night, rating: ratingBinding(for: night.id), predicted: aiPrediction(night))
                        } label: {
                            nightRow(night)
                        }
                        .listRowBackground(Theme.card)
                    }
                } header: {
                    HStack {
                        Text("Week of \(ws[index].start.formatted(.dateTime.day().month()))")
                            .font(Theme.serif(17))
                            .foregroundStyle(Theme.text)
                        Spacer()
                        Text(weekSummary(ws[index].nights))
                            .font(.footnote)
                            .foregroundStyle(weekColor(ws[index].nights))
                    }
                    .textCase(nil)
                }
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .overlay {
            if isLoading { ProgressView("Reading Apple Health...") }
            else if let loadError { ContentUnavailableView("Health data unavailable", systemImage: "heart.slash", description: Text(loadError)) }
            else if nights.isEmpty { ContentUnavailableView("No overnight data", systemImage: "bed.double", description: Text("Check Health permissions and make sure your iPhone or Apple Watch has recorded sleep and heart measurements.")) }
        }
        .refreshable { await reload() }
    }

    private var gymTab: some View {
        let calendar: Calendar = {
            var c = Calendar(identifier: .gregorian)
            c.firstWeekday = 2
            return c
        }()
        let monday = calendar.dateInterval(of: .weekOfYear, for: Date())?.start ?? Date()
        let thisWeek = workouts.filter { $0.id >= monday }
        let minutes = thisWeek.map(\.minutes).reduce(0, +) / 60
        let letters = ["M", "T", "W", "T", "F", "S", "S"]
        return List {
            VStack(spacing: 14) {
                HStack(spacing: 10) {
                    ForEach(0..<7, id: \.self) { offset in
                        let day = calendar.date(byAdding: .day, value: offset, to: monday) ?? monday
                        let trained = workouts.contains { calendar.isDate($0.id, inSameDayAs: day) }
                        VStack(spacing: 6) {
                            Circle()
                                .fill(trained ? Theme.text : Color.clear)
                                .overlay(Circle().stroke(trained ? Theme.text : Theme.muted.opacity(0.4), lineWidth: 1.5))
                                .frame(width: 34, height: 34)
                            Text(letters[offset]).font(.caption).foregroundStyle(Theme.muted)
                        }
                    }
                }
                Text("\(thisWeek.count) sessions this week, \(hoursAndMinutes(minutes)) under the bar")
                    .font(Theme.serif(18))
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)

            ForEach(workouts) { workout in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(workout.id.formatted(.dateTime.weekday(.wide)))
                        Text(workout.id.formatted(.dateTime.day().month())).font(.caption).foregroundStyle(Theme.muted)
                    }
                    Spacer()
                    Text("\(Int(workout.minutes.rounded())) min").font(Theme.serif(17))
                    Label("\(bpmText(workout.averageHeartRate)) / \(bpmText(workout.peakHeartRate))", systemImage: "heart.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.stop)
                }
                .listRowBackground(Theme.card)
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Theme.bg)
        .overlay {
            if !isLoading && workouts.isEmpty {
                ContentUnavailableView("No strength workouts", systemImage: "dumbbell", description: Text("Strength workouts recorded in Apple Health will appear here."))
            }
        }
        .refreshable { await reload() }
    }

    private var wakeDate: Binding<Date> {
        Binding(
            get: { Calendar.current.date(bySettingHour: wakeMinutes / 60, minute: wakeMinutes % 60, second: 0, of: Date()) ?? Date() },
            set: { newValue in
                let parts = Calendar.current.dateComponents([.hour, .minute], from: newValue)
                wakeMinutes = (parts.hour ?? 7) * 60 + (parts.minute ?? 0)
            }
        )
    }

    private func bedtime(for hours: Double) -> String {
        let minutes = ((wakeMinutes - Int((hours * 60).rounded())) % 1440 + 1440) % 1440
        return String(format: "%02d:%02d", minutes / 60, minutes % 60)
    }

    private var sleepNeedTab: some View {
        let target = sleepTarget
        return ScrollView {
            VStack(spacing: 20) {
                VStack(spacing: 6) {
                    Text("Sleep tonight").font(.subheadline).foregroundStyle(Theme.muted)
                    Text(hoursAndMinutes(target.hours))
                        .font(Theme.serif(76, .bold))
                        .foregroundStyle(Theme.text)
                        .contentTransition(.numericText())
                    Text("Be asleep by \(bedtime(for: target.hours))")
                        .font(Theme.serif(20))
                        .foregroundStyle(Theme.caution)
                    Text(target.isPersonalised ? "Tuned to your ratings" : "Starting estimate")
                        .font(.caption)
                        .foregroundStyle(Theme.muted)
                }
                .padding(.top, 8)

                Card {
                    HStack {
                        Label("I wake up at", systemImage: "alarm.fill")
                        Spacer()
                        DatePicker("Wake time", selection: wakeDate, displayedComponents: .hourAndMinute)
                            .labelsHidden()
                    }
                }

                Card {
                    VStack(spacing: 12) {
                        LabeledContent("Usual sleep need", value: hoursAndMinutes(target.baseline))
                        LabeledContent("Sleep debt, last 7 days", value: hoursAndMinutes(target.debt))
                        LabeledContent("Extra tonight for debt", value: hoursAndMinutes(min(1.5, target.debt * 0.25)))
                        LabeledContent("Today's training", value: String(format: "%.1f h", target.training))
                        Divider()
                        LabeledContent("Rated mornings", value: "\(target.samples) of 12")
                    }
                }

                Text(target.isPersonalised
                     ? "Your model is learning how sleep, training and recent sleep debt relate to your morning ratings."
                     : target.samples >= 12
                        ? "Your ratings don't show a clear best sleep length yet, so this is a starting estimate."
                        : "Rate \(max(0, 12 - target.samples)) more mornings to start tuning this to you.")
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
                    .multilineTextAlignment(.center)

                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("On-device AI", systemImage: "sparkles")
                            .font(Theme.serif(18))
                            .foregroundStyle(Theme.ai)
                        Text(ai.status).font(.subheadline).foregroundStyle(Theme.muted)
                        if let hours = aiSleep {
                            Text("AI suggests \(hoursAndMinutes(hours)) tonight").font(.headline)
                        }
                        Button("Retrain now") { Task { await ai.train(aiExamples) } }
                            .buttonStyle(.bordered)
                            .tint(Theme.ai)
                    }
                }

                Text("Wellness estimate, not medical advice.")
                    .font(.caption2)
                    .foregroundStyle(Theme.muted)
            }
            .padding()
        }
        .background(Theme.bg)
        .sensoryFeedback(.success, trigger: ai.trainedOn)
    }

    var body: some View {
        TabView {
            NavigationStack { sleepTab.navigationTitle(greeting) }
                .tabItem { Label("Sleep", systemImage: "bed.double.fill") }
            NavigationStack { gymTab.navigationTitle("Gym") }
                .tabItem { Label("Gym", systemImage: "dumbbell.fill") }
            NavigationStack { sleepNeedTab.navigationTitle("Tonight") }
                .tabItem { Label("Need", systemImage: "moon.zzz.fill") }
        }
        .tint(Theme.text)
        .preferredColorScheme(.dark)
        .task { await reload() }
    }

    @MainActor
    private func reload() async {
        isLoading = true
        loadError = nil
        try? await health.requestAccess()   // ask for permission before any query runs
        async let loadedNights = health.loadNights(days: 90)
        async let loadedWorkouts = health.loadStrengthWorkouts(days: 90)
        nights = await loadedNights
        workouts = await loadedWorkouts
        isLoading = false
        if nights.isEmpty && workouts.isEmpty {
            loadError = "No readable HealthKit records were returned. Check Settings, Privacy & Security, Health, and your app."
        }
        await ai.train(aiExamples)
    }
}
