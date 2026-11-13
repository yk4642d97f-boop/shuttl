import Foundation
import ShuttlCore
import ShuttlDetector

setvbuf(stdout, nil, _IOLBF, 0)

struct Args {
    var video = "", labels = "", out = ""
    var models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../../../iOS/Packages/ShuttlDetector/Models").standardized.path
    var poseStep = 1
    var maxRallies = Int.max
    var calFrames = 24
    var pad = 2.0
    var padAfter = 6.0
    var cache = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../.mlcache").standardized.path

    init(_ argv: [String]) {
        var it = argv.dropFirst().makeIterator()
        while let a = it.next() {
            let v = it.next() ?? ""
            switch a {
            case "--video": video = v
            case "--labels": labels = v
            case "--out": out = v
            case "--models": models = v
            case "--pose-step": poseStep = Int(v) ?? 1
            case "--max-rallies": maxRallies = Int(v) ?? .max
            case "--cal-frames": calFrames = Int(v) ?? 24
            case "--cache": cache = v
            case "--pad": pad = Double(v) ?? 2.0
            case "--pad-after": padAfter = Double(v) ?? 6.0
            default: fatalError("unknown argument \(a)")
            }
        }
        if video.isEmpty || labels.isEmpty || out.isEmpty {
            print("usage: extract --video V.mp4 --labels SET_CSV_DIR --out OUT.npz [--pose-step 1] [--max-rallies N] [--models DIR]")
            exit(2)
        }
    }
}

func log(_ s: String) {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss"
    print("[\(f.string(from: Date()))] \(s)")
}

let args = Args(CommandLine.arguments)
let outURL = URL(fileURLWithPath: args.out)
let parts = outURL.deletingPathExtension().appendingPathExtension("parts")
try FileManager.default.createDirectory(at: parts, withIntermediateDirectories: true)
let plist = PropertyListEncoder()
plist.outputFormat = .binary

func load<T: Decodable>(_ type: T.Type, _ url: URL) -> T? {
    (try? Data(contentsOf: url)).flatMap { try? PropertyListDecoder().decode(type, from: $0) }
}

let wall0 = CFAbsoluteTimeGetCurrent()
let video = try await VideoSource.open(URL(fileURLWithPath: args.video))
let meta = video.meta
let rallies = Array(try Labels.rallies(in: URL(fileURLWithPath: args.labels), fps: meta.fps,
                                       nFrames: meta.nFrames, padBefore: args.pad,
                                       padAfter: args.padAfter).prefix(args.maxRallies))
let spanFrames = rallies.reduce(0) { $0 + $1.stop - $1.start }
log("\(args.video): \(meta.nFrames) frames @ \(meta.fps) fps \(meta.width)x\(meta.height), \(rallies.count) rallies, \(spanFrames) span frames, probe \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - wall0)) s")

let matchCalURL = parts.appendingPathComponent("match_court.plist")
let matchCourt: CourtResult
if let c = load(CourtResult.self, matchCalURL) {
    matchCourt = c
} else {
    let hits = rallies.flatMap(\.hitFrames).filter { $0 < meta.nFrames }.sorted()
    let idx = Array(Set((0..<args.calFrames).map { hits[$0 * (hits.count - 1) / max(args.calFrames - 1, 1)] })).sorted()
    matchCourt = Court.calibrate(try Court.frames(video, idx), indices: idx, size: meta.size)
    try plist.encode(matchCourt).write(to: matchCalURL)
}
log("match court: \(matchCourt.reason) conf \(matchCourt.confidence.map { String(format: "%.2f", $0) } ?? "-") in \(String(format: "%.1f", matchCourt.seconds)) s")

let detector = try await HeatmapDetector.build(modelsDirectory: URL(fileURLWithPath: args.models),
                                               cacheDirectory: URL(fileURLWithPath: args.cache))
let extractor = RallyExtractor(video: video, detector: detector, matchCamera: matchCourt.camera, poseStep: args.poseStep)

var results: [RallyOut] = []
var doneFrames = 0, newFrames = 0
let run0 = CFAbsoluteTimeGetCurrent()
for (i, spec) in rallies.enumerated() {
    let url = parts.appendingPathComponent(spec.key + ".plist")
    var r: RallyOut
    if let cached = load(RallyOut.self, url), cached.spec.start == spec.start, cached.spec.stop == spec.stop {
        r = cached
    } else {
        let t = CFAbsoluteTimeGetCurrent()
        do { r = try extractor.run(spec) } catch {
            r = RallyOut(spec: spec, error: String(describing: error))
        }
        r.timing["total"] = CFAbsoluteTimeGetCurrent() - t
        try plist.encode(r).write(to: url, options: .atomic)
        newFrames += r.n
        let fps = Double(newFrames) / (CFAbsoluteTimeGetCurrent() - run0)
        let stages = ["calibrate", "detect", "link", "pose", "track"].map { k in "\(k) \(String(format: "%.1f", r.timing[k] ?? .nan))" }
        log("rally \(i + 1)/\(rallies.count) \(spec.key) \(r.n) fr: \(stages.joined(separator: " ")) s; cam \(r.cameraSource) "
            + "\(r.court?.reason ?? "-") Δ\(String(format: "%.1f", r.cameraDiffPx)) px; \(String(format: "%.1f", fps)) fr/s"
            + (r.error.map { " ERROR \($0)" } ?? ""))
    }
    doneFrames += r.n
    results.append(r)
}

try Output.write(results, matchCourt: matchCourt, meta: meta, args: args, detector: detector.spec.name, to: outURL)
log("wrote \(outURL.path): \(doneFrames) frames, \(results.filter { $0.error != nil }.count) failed rallies, \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - wall0)) s")
