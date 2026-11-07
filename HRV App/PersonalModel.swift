import CoreML
import Combine

/// Inputs: HRV z, sleeping HR z, breathing score, sleep hours minus 8,
/// training hours the day before, sleep debt in hours divided by 2.
/// Must match the order in build_model.py.
func modelInput(_ n: Night, nights: [Night], workouts: [WorkoutSession], baseline: Double) -> [Double]? {
    guard let f = n.features, f.count >= 3 else { return nil }
    let cal = Calendar.current
    let prev = cal.date(byAdding: .day, value: -1, to: n.id) ?? n.id
    let train = workouts.filter { cal.isDate($0.id, inSameDayAs: prev) }.map(\.minutes).reduce(0, +) / 60
    let start = cal.date(byAdding: .day, value: -7, to: n.id) ?? n.id
    let debt = nights.filter { $0.id >= start && $0.id < n.id }.reduce(0.0) { $0 + max(0, baseline - $1.sleepHours) }
    return [f[0], f[1], f[2], n.sleepHours - 8, min(train, 3), min(debt, 5) / 2]
}

/// Keeps a running update task alive until it finishes, then releases it.
private final class TaskBox { var task: MLUpdateTask? }

@MainActor
final class PersonalModel: ObservableObject {
    @Published var status = "Starting model"
    @Published var trainedOn = 0
    private var model: MLModel?

    private let config: MLModelConfiguration = {
        let c = MLModelConfiguration()
        c.computeUnits = .cpuAndNeuralEngine   // iOS decides what actually runs where
        return c
    }()

    private var savedURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("PersonalModel.mlmodelc")
    }
    private var bundledURL: URL? { Bundle.main.url(forResource: "RatingModel", withExtension: "mlmodelc") }

    init() { load() }

    func load() {
        let saved = FileManager.default.fileExists(atPath: savedURL.path)
        guard let url = saved ? savedURL : bundledURL else {
            status = "RatingModel.mlmodel is missing from the app target"
            return
        }
        model = try? MLModel(contentsOf: url, configuration: config)
        trainedOn = UserDefaults.standard.integer(forKey: "aiTrainedOn")
        status = trainedOn > 0 ? "Personalized on \(trainedOn) ratings" : "Starting model (copies the built-in formula)"
    }

    /// Predicted morning rating, 1 to 5
    func predict(_ x: [Double]) -> Double? {
        guard let model, x.count == 6, let a = try? MLMultiArray(shape: [6], dataType: .double) else { return nil }
        for i in 0..<6 { a[i] = NSNumber(value: x[i]) }
        guard let out = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": a])),
              let r = out.featureValue(for: "rating")?.multiArrayValue else { return nil }
        return min(5, max(1, r[0].doubleValue))
    }

    /// Shortest sleep between 7 and 10 h whose predicted rating is within 0.05 of the best,
    /// for an average night with today's training and current debt
    func bestSleep(training: Double, debt: Double) -> Double? {
        var options: [(Double, Double)] = []
        for minutes in stride(from: 420, through: 600, by: 15) {
            let h = Double(minutes) / 60
            if let p = predict([0, 0, 0, h - 8, min(training, 3), min(debt, 5) / 2]) { options.append((h, p)) }
        }
        guard let top = options.map({ $0.1 }).max() else { return nil }
        return options.first { $0.1 >= top - 0.05 }?.0
    }

    /// Retrains from the bundled starting model on all your rated mornings
    func train(_ examples: [([Double], Double)]) async {
        guard examples.count >= 10, let src = bundledURL else { return }
        status = "Training on your phone..."
        let dest = savedURL
        let cfg = config
        do {
            var items: [MLFeatureProvider] = []
            for (x, y) in examples {
                let a = try MLMultiArray(shape: [6], dataType: .double)
                for i in 0..<6 { a[i] = NSNumber(value: x[i]) }
                let t = try MLMultiArray(shape: [1], dataType: .double)
                t[0] = NSNumber(value: y)
                items.append(try MLDictionaryFeatureProvider(dictionary: ["features": a, "rating_true": t]))
            }
            let batch = MLArrayBatchProvider(array: items)

            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                let box = TaskBox()
                do {
                    box.task = try MLUpdateTask(forModelAt: src, trainingData: batch, configuration: cfg) { ctx in
                        defer { box.task = nil }
                        if let e = ctx.task.error { c.resume(throwing: e); return }
                        do {
                            let tmp = dest.deletingLastPathComponent().appendingPathComponent("PersonalModelTmp.mlmodelc")
                            try? FileManager.default.removeItem(at: tmp)
                            try ctx.model.write(to: tmp)
                            try? FileManager.default.removeItem(at: dest)
                            try FileManager.default.moveItem(at: tmp, to: dest)
                            c.resume()
                        } catch { c.resume(throwing: error) }
                    }
                    box.task?.resume()
                } catch { c.resume(throwing: error) }
            }

            model = try MLModel(contentsOf: dest, configuration: cfg)
            trainedOn = examples.count
            UserDefaults.standard.set(trainedOn, forKey: "aiTrainedOn")
            let miss = examples.map { abs((predict($0.0) ?? $0.1) - $0.1) }.reduce(0, +) / Double(examples.count)
            status = "Personalized on \(trainedOn) ratings (avg miss \(String(format: "%.1f", miss)) on your own data)"
        } catch {
            status = "Training failed: \(error.localizedDescription)"
        }
    }
}
