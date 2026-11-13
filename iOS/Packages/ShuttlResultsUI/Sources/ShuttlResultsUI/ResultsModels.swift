import Foundation
import simd
import ShuttlCore

// MARK: - Public result vocabulary

public extension ShotType {
    var chineseName: String {
        switch self {
        case .smash: return "扣杀"
        case .clear: return "高远"
        case .drop: return "吊球"
        case .net: return "网前"
        case .drive: return "平抽"
        case .serve: return "发球"
        case .lift: return "挑球"
        case .push: return "推球"
        case .kill: return "杀球"
        case .block: return "挡网"
        case .crossNet: return "勾球"
        case .shortServe: return "发短球"
        case .longServe: return "发长球"
        case .unknown: return "其他"
        }
    }

    var resultsInkIndex: Int {
        switch self {
        case .smash: return 0
        case .clear: return 1
        case .drop: return 2
        case .net: return 3
        case .drive: return 4
        case .serve: return 5
        case .lift: return 6
        case .push: return 7
        case .kill: return 8
        case .block: return 9
        case .unknown: return 10
        case .crossNet: return 11
        case .shortServe: return 12
        case .longServe: return 13
        }
    }
}

public extension RallyEndReason {
    var chineseName: String {
        switch self {
        case .inCourt: return "界内"
        case .out: return "出界"
        case .net: return "触网"
        case .unknown: return "未知"
        }
    }
}

public extension CourtSide {
    var chineseName: String {
        switch self {
        case .near: return "近端"
        case .far: return "远端"
        case .unknown: return "未判定"
        }
    }
}

public enum ResultsSideFilter: String, CaseIterable, Sendable {
    case all, near, far

    public var chineseName: String {
        switch self {
        case .all: return "全部"
        case .near: return "近端"
        case .far: return "远端"
        }
    }

    fileprivate func includes(_ side: CourtSide) -> Bool {
        switch self {
        case .all: return true
        case .near: return side == .near
        case .far: return side == .far
        }
    }
}

public struct ResultMetric: Sendable, Equatable {
    public let value: Double?
    public let unit: String?
    public let isAvailable: Bool

    public init(value: Double?, unit: String? = nil, isAvailable: Bool? = nil) {
        self.value = value?.isFinite == true ? value : nil
        self.unit = unit
        self.isAvailable = isAvailable ?? (self.value != nil)
    }

    public var displayValue: String {
        guard let value, isAvailable else { return "--" }
        return String(format: "%.1f", value)
    }
}

public struct MatchSummary: Sendable, Equatable {
    public let totalRallies: Int
    public let totalShots: Int
    public let maxSpeedKmh: Double?
    public let averageSpeedKmh: Double?
    public let maxSmashKmh: Double?
    public let averageSmashKmh: Double?
    public let averageRallyShots: Double
    public let shotTypeCounts: [ShotType: Int]
    public let endReasonCounts: [RallyEndReason: Int]
    public let pointsWon: [CourtSide: Int]
    public let hasMetricData: Bool

    public init(totalRallies: Int, totalShots: Int, maxSpeedKmh: Double?, averageSpeedKmh: Double?,
                maxSmashKmh: Double?, averageSmashKmh: Double?, averageRallyShots: Double,
                shotTypeCounts: [ShotType: Int], endReasonCounts: [RallyEndReason: Int],
                pointsWon: [CourtSide: Int], hasMetricData: Bool) {
        self.totalRallies = totalRallies
        self.totalShots = totalShots
        self.maxSpeedKmh = maxSpeedKmh
        self.averageSpeedKmh = averageSpeedKmh
        self.maxSmashKmh = maxSmashKmh
        self.averageSmashKmh = averageSmashKmh
        self.averageRallyShots = averageRallyShots
        self.shotTypeCounts = shotTypeCounts
        self.endReasonCounts = endReasonCounts
        self.pointsWon = pointsWon
        self.hasMetricData = hasMetricData
    }

    public init(analysis: MatchAnalysis) {
        let shots = analysis.rallies.flatMap(\.shots)
        let speeds = analysis.camera == nil ? [] : shots.compactMap { shot -> Double? in
            guard let segment = shot.segment else { return nil }
            let speed = segment.speed0Kmh
            return speed.isFinite && speed >= 0 ? speed : nil
        }
        let smashes = analysis.camera == nil ? [] : shots.compactMap { shot -> Double? in
            guard shot.shotType == .smash || shot.shotType == .kill,
                  let segment = shot.segment else { return nil }
            let speed = segment.speed0Kmh
            return speed.isFinite && speed >= 0 ? speed : nil
        }
        var shotCounts: [ShotType: Int] = [:]
        shots.forEach { shotCounts[$0.shotType, default: 0] += 1 }
        var endCounts: [RallyEndReason: Int] = [:]
        analysis.rallies.forEach { endCounts[$0.endReason, default: 0] += 1 }
        var points: [CourtSide: Int] = [.near: 0, .far: 0, .unknown: 0]
        analysis.rallies.forEach { points[$0.winner, default: 0] += 1 }

        self.init(
            totalRallies: analysis.rallies.count,
            totalShots: shots.count,
            maxSpeedKmh: speeds.max()?.rounded(toPlaces: 1),
            averageSpeedKmh: speeds.isEmpty ? nil : (speeds.reduce(0, +) / Double(speeds.count)).rounded(toPlaces: 1),
            maxSmashKmh: smashes.max()?.rounded(toPlaces: 1),
            averageSmashKmh: smashes.isEmpty ? nil : (smashes.reduce(0, +) / Double(smashes.count)).rounded(toPlaces: 1),
            averageRallyShots: (analysis.rallies.isEmpty ? 0 : Double(shots.count) / Double(analysis.rallies.count)).rounded(toPlaces: 1),
            shotTypeCounts: shotCounts,
            endReasonCounts: endCounts,
            pointsWon: points,
            hasMetricData: analysis.camera != nil && !speeds.isEmpty
        )
    }
}

public struct ScoreStep: Identifiable, Sendable, Equatable {
    public let id: Int
    public let rallyIndex: Int
    public let near: Int
    public let far: Int
    public let winner: CourtSide
    public let isUncertain: Bool

    public init(id: Int, rallyIndex: Int, near: Int, far: Int, winner: CourtSide, isUncertain: Bool) {
        self.id = id
        self.rallyIndex = rallyIndex
        self.near = near
        self.far = far
        self.winner = winner
        self.isUncertain = isUncertain
    }
}

public struct ShotPoint: Identifiable, Sendable, Equatable {
    public let id: Int
    public let rallyIndex: Int
    public let shotIndex: Int
    public let side: CourtSide
    public let shotType: ShotType
    public let point: SIMD2<Double>
    public let isIn: Bool
    public let marginM: Double?
    public let confidence: Double

    public init(id: Int, rallyIndex: Int, shotIndex: Int, side: CourtSide, shotType: ShotType,
                point: SIMD2<Double>, isIn: Bool, marginM: Double?, confidence: Double) {
        self.id = id
        self.rallyIndex = rallyIndex
        self.shotIndex = shotIndex
        self.side = side
        self.shotType = shotType
        self.point = point
        self.isIn = isIn
        self.marginM = marginM
        self.confidence = confidence
    }
}

public struct FlightSummary: Sendable, Equatable {
    public let startFrame: Int
    public let endFrame: Int
    public let initialPositionM: SIMD3<Double>
    public let initialVelocityMS: SIMD3<Double>
    public let speedKmh: Double?
    public let apexM: Double?
    public let netClearanceM: Double?
    public let netCrossingY: Double?
    public let landingXY: SIMD2<Double>?
    public let landingT: Double?
    public let rmsePx: Double?
    public let confidence: Double
    public let frames: [Int]
    public let pathNormalized: [SIMD2<Double>]
    public let endReason: SegmentEndReason

    public init(startFrame: Int, endFrame: Int,
                initialPositionM: SIMD3<Double> = SIMD3(.nan, .nan, .nan),
                initialVelocityMS: SIMD3<Double> = SIMD3(.nan, .nan, .nan),
                speedKmh: Double?, apexM: Double?, netClearanceM: Double?,
                netCrossingY: Double?, landingXY: SIMD2<Double>?, landingT: Double?,
                rmsePx: Double?, confidence: Double, frames: [Int] = [],
                pathNormalized: [SIMD2<Double>] = [], endReason: SegmentEndReason) {
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.initialPositionM = initialPositionM
        self.initialVelocityMS = initialVelocityMS
        self.speedKmh = speedKmh
        self.apexM = apexM
        self.netClearanceM = netClearanceM
        self.netCrossingY = netCrossingY
        self.landingXY = landingXY
        self.landingT = landingT
        self.rmsePx = rmsePx
        self.confidence = confidence
        self.frames = frames
        self.pathNormalized = pathNormalized
        self.endReason = endReason
    }
}

public struct ShotRow: Identifiable, Sendable, Equatable {
    public let id: Int
    public let index: Int
    public let side: CourtSide
    public let shotType: ShotType
    public let legacyShotType: ShotType
    public let shotTypeConfidence: Double
    public let strokeSide: StrokeSide
    public let time: Double
    public let pixelNorm: SIMD2<Double>
    public let hitterPosition: SIMD2<Double>?
    public let speedKmh: Double?
    public let heightM: Double?
    public let frame: Int
    public let confidence: Double
    public let flight: FlightSummary?

    public init(id: Int, index: Int, side: CourtSide, shotType: ShotType,
                legacyShotType: ShotType = .unknown, shotTypeConfidence: Double = 0,
                strokeSide: StrokeSide = .unknown, time: Double = .nan,
                pixelNorm: SIMD2<Double> = SIMD2(.nan, .nan), hitterPosition: SIMD2<Double>? = nil,
                speedKmh: Double?, heightM: Double?, frame: Int, confidence: Double,
                flight: FlightSummary? = nil) {
        self.id = id
        self.index = index
        self.side = side
        self.shotType = shotType
        self.legacyShotType = legacyShotType
        self.shotTypeConfidence = shotTypeConfidence
        self.strokeSide = strokeSide
        self.time = time
        self.pixelNorm = pixelNorm
        self.hitterPosition = hitterPosition
        self.speedKmh = speedKmh
        self.heightM = heightM
        self.frame = frame
        self.confidence = confidence
        self.flight = flight
    }
}

public struct RallySummary: Identifiable, Sendable, Equatable {
    public let id: Int
    public let index: Int
    public let startFrame: Int
    public let endFrame: Int
    public let shots: [ShotRow]
    public let endReason: RallyEndReason
    public let winner: CourtSide
    public let server: CourtSide
    public let landingXY: SIMD2<Double>?
    public let inOutMarginM: Double?
    public let confidence: Double
    public let segmentationConfidence: Double
    public let outcomeConfidence: Double
    public let outcomeSource: String
    public let serveMissed: Bool

    public init(id: Int, index: Int, startFrame: Int, endFrame: Int, shots: [ShotRow],
                endReason: RallyEndReason, winner: CourtSide, server: CourtSide,
                landingXY: SIMD2<Double>?, inOutMarginM: Double? = nil, confidence: Double,
                segmentationConfidence: Double = 0, outcomeConfidence: Double,
                outcomeSource: String = "unknown", serveMissed: Bool) {
        self.id = id
        self.index = index
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.shots = shots
        self.endReason = endReason
        self.winner = winner
        self.server = server
        self.landingXY = landingXY
        self.inOutMarginM = inOutMarginM
        self.confidence = confidence
        self.segmentationConfidence = segmentationConfidence
        self.outcomeConfidence = outcomeConfidence
        self.outcomeSource = outcomeSource
        self.serveMissed = serveMissed
    }
}

public struct PlayerMovementStats: Sendable, Equatable {
    public let distanceM: Double
    public let averageSpeedMS: Double
    public let maxSpeedMS: Double
    public let heatmap: [[Double]]
    public let courtCoverage: Double
    public let trackedSeconds: Double
    fileprivate let dwellSeconds: [[Double]]

    public init(distanceM: Double = 0, averageSpeedMS: Double = 0, maxSpeedMS: Double = 0,
                heatmap: [[Double]] = Array(repeating: Array(repeating: 0, count: 10), count: 12),
                courtCoverage: Double = 0, trackedSeconds: Double = 0) {
        self.distanceM = distanceM
        self.averageSpeedMS = averageSpeedMS
        self.maxSpeedMS = maxSpeedMS
        self.heatmap = heatmap
        self.courtCoverage = courtCoverage
        self.trackedSeconds = trackedSeconds
        self.dwellSeconds = Array(repeating: Array(repeating: 0, count: 10), count: 12)
    }

    fileprivate init(distanceM: Double, averageSpeedMS: Double, maxSpeedMS: Double,
                     heatmap: [[Double]], courtCoverage: Double, trackedSeconds: Double,
                     dwellSeconds: [[Double]]) {
        self.distanceM = distanceM
        self.averageSpeedMS = averageSpeedMS
        self.maxSpeedMS = maxSpeedMS
        self.heatmap = heatmap
        self.courtCoverage = courtCoverage
        self.trackedSeconds = trackedSeconds
        self.dwellSeconds = dwellSeconds
    }
}

public struct MovementStats: Sendable, Equatable {
    public let distanceM: Double
    public let averageSpeedMS: Double
    public let maxSpeedMS: Double
    public let heatmap: [[Double]]
    public let courtCoverage: Double
    public let trackedSeconds: Double
    public let playerStats: [Int: PlayerMovementStats]

    public init(distanceM: Double = 0, averageSpeedMS: Double = 0, maxSpeedMS: Double = 0,
                heatmap: [[Double]] = Array(repeating: Array(repeating: 0, count: 10), count: 12),
                courtCoverage: Double = 0, trackedSeconds: Double = 0,
                playerStats: [Int: PlayerMovementStats] = [:]) {
        self.distanceM = distanceM
        self.averageSpeedMS = averageSpeedMS
        self.maxSpeedMS = maxSpeedMS
        self.heatmap = heatmap
        self.courtCoverage = courtCoverage
        self.trackedSeconds = trackedSeconds
        self.playerStats = playerStats
    }
}

public struct ResultsNotice: Identifiable, Sendable, Equatable {
    public enum Kind: String, Sendable { case calibration, caution, confidence, empty }
    public let id: String
    public let kind: Kind
    public let title: String
    public let message: String
    /// Several short lines instead of one message (the caution note).
    public let lines: [String]

    public init(id: String, kind: Kind, title: String, message: String, lines: [String] = []) {
        self.id = id
        self.kind = kind
        self.title = title
        self.message = message
        self.lines = lines
    }
}

public struct MatchResultViewModel: Sendable {
    public let analysis: MatchAnalysis
    public let summary: MatchSummary
    public let scoreProgression: [ScoreStep]
    public let rallies: [RallySummary]
    public let landingPoints: [ShotPoint]
    public let notices: [ResultsNotice]
    public let movement: [CourtSide: MovementStats]
    public let isCalibrated: Bool

    /// - cautions: "results are shown, but…" lines (camera bumped, landings withdrawn,
    ///   far end seen too flat), in the order given. They share one quiet paper note.
    public init(analysis: MatchAnalysis, cautions: [String] = []) {
        self.analysis = analysis
        self.summary = MatchSummary(analysis: analysis)
        self.isCalibrated = analysis.camera != nil
        self.scoreProgression = Self.makeScoreProgression(analysis.score, rallies: analysis.rallies)
        self.rallies = analysis.rallies.enumerated().map { rallyIndex, rally in
            RallySummary(
                id: rallyIndex,
                index: rallyIndex + 1,
                startFrame: rally.startFrame,
                endFrame: rally.endFrame,
                shots: rally.shots.enumerated().map { shotIndex, shot in
                    ShotRow(
                        id: rallyIndex * 10_000 + shotIndex,
                        index: shotIndex + 1,
                        side: shot.hit.hitter,
                        shotType: shot.shotType,
                        legacyShotType: shot.shotType.legacy,
                        shotTypeConfidence: shot.shotTypeConfidence,
                        strokeSide: shot.strokeSide,
                        time: shot.hit.t,
                        pixelNorm: Self.normalizedPixel(shot.hit.xyPx, meta: analysis.meta),
                        hitterPosition: shot.hitterPosWorld?.isFinite == true ? shot.hitterPosWorld : nil,
                        speedKmh: analysis.camera == nil ? nil : Self.speed(of: shot),
                        heightM: shot.hitHeightM?.isFinite == true ? shot.hitHeightM : nil,
                        frame: shot.hit.frame,
                        confidence: shot.hit.confidence,
                        flight: analysis.camera == nil ? nil : Self.flight(of: shot.segment)
                    )
                },
                endReason: rally.endReason,
                winner: rally.winner,
                server: rally.server,
                landingXY: rally.landingXY?.isFinite == true ? rally.landingXY : nil,
                inOutMarginM: rally.inOutMarginM,
                confidence: rally.confidence,
                segmentationConfidence: rally.segmentationConfidence,
                outcomeConfidence: rally.outcomeConfidence,
                outcomeSource: rally.outcomeSource,
                serveMissed: rally.serveMissed
            )
        }
        self.landingPoints = Self.makeLandingPoints(analysis)
        self.movement = [
            .near: Self.movementStats(analysis.players, side: .near, meta: analysis.meta),
            .far: Self.movementStats(analysis.players, side: .far, meta: analysis.meta),
        ]
        var notices = Self.makeNotices(analysis, summary: summary)
        var lines = cautions.filter { !$0.isEmpty }
        if !lines.isEmpty {
            // One "留意" card, not two near-identical boxes: low confidence becomes its last line.
            if let i = notices.firstIndex(where: { $0.kind == .confidence }) {
                lines.append(notices.remove(at: i).message)
            }
            notices.insert(ResultsNotice(id: "caution", kind: .caution, title: "结果可以看，但要留意",
                                         message: "", lines: lines), at: 0)
        }
        self.notices = notices
    }

    /// The shared "results are shown, but…" note (nil when there is nothing to say).
    public var cautionNotice: ResultsNotice? { notices.first { $0.kind == .caution } }

    public func filteredLandingPoints(side: ResultsSideFilter = .all, shotType: ShotType? = nil) -> [ShotPoint] {
        landingPoints.filter { side.includes($0.side) && (shotType == nil || $0.shotType == shotType) }
    }

    public func rally(at index: Int) -> RallySummary? {
        guard rallies.indices.contains(index) else { return nil }
        return rallies[index]
    }

    private static func speed(of shot: ShotEventV2) -> Double? {
        guard let segment = shot.segment else { return nil }
        let value = segment.speed0Kmh
        return value.isFinite && value >= 0 ? value.rounded(toPlaces: 1) : nil
    }

    private static func normalizedPixel(_ pixel: SIMD2<Double>, meta: VideoMeta) -> SIMD2<Double> {
        guard pixel.isFinite, meta.width > 0, meta.height > 0 else { return SIMD2(.nan, .nan) }
        return SIMD2(pixel.x / Double(meta.width), pixel.y / Double(meta.height))
    }

    private static func flight(of segment: ShotSegment3D?) -> FlightSummary? {
        guard let segment else { return nil }
        let apex = segment.positions.compactMap { $0.z.isFinite ? $0.z : nil }.max()
        return FlightSummary(startFrame: segment.startFrame, endFrame: segment.endFrame,
                             initialPositionM: segment.p0, initialVelocityMS: segment.v0,
                             speedKmh: speed(of: segment), apexM: apex,
                             netClearanceM: segment.netCrossingHeight.map { $0 - CourtModel.netHeightCentre },
                             netCrossingY: segment.netCrossingY,
                             landingXY: segment.landingXY?.isFinite == true ? segment.landingXY : nil,
                             landingT: segment.landingT, rmsePx: segment.rmsePx.isFinite ? segment.rmsePx : nil,
                             confidence: segment.confidence, frames: segment.frames,
                             pathNormalized: [], endReason: segment.endReason)
    }

    private static func speed(of segment: ShotSegment3D) -> Double? {
        let value = segment.speed0Kmh
        return value.isFinite && value >= 0 ? value.rounded(toPlaces: 1) : nil
    }

    private static func makeScoreProgression(_ values: [JSONValue], rallies: [RallyV2]) -> [ScoreStep] {
        values.enumerated().compactMap { index, value in
            guard case .object(let object) = value else { return nil }
            let score = object["score"]?.objectValue ?? [:]
            let near = score["near"]?.intValue ?? 0
            let far = score["far"]?.intValue ?? 0
            let winner = rallies.indices.contains(index) ? rallies[index].winner : .unknown
            return ScoreStep(id: index, rallyIndex: index, near: near, far: far, winner: winner,
                             isUncertain: object["uncertain"]?.boolValue == true)
        }
    }

    private static func makeLandingPoints(_ analysis: MatchAnalysis) -> [ShotPoint] {
        guard analysis.camera != nil else { return [] }
        return analysis.rallies.enumerated().compactMap { rallyIndex, rally in
            // The rally-level landing is authoritative for an out/net decision;
            // a terminal segment may omit landing_xy when it stopped at the net.
            let candidate = rally.landingXY ?? rally.shots.last?.segment?.landingXY
            guard let landing = candidate, landing.isFinite else { return nil }
            let shotIndex = max(0, rally.shots.count - 1)
            let shot = rally.shots.indices.contains(shotIndex) ? rally.shots[shotIndex] : nil
            let margin = CourtModel.inOutMargin(x: landing.x, y: landing.y, doubles: true)
            let isIn = rally.endReason == .inCourt || (rally.endReason == .unknown && margin >= 0)
            let confidence = min(rally.outcomeConfidence > 0 ? rally.outcomeConfidence : rally.confidence,
                                 shot?.hit.confidence ?? 1)
            return ShotPoint(id: rallyIndex * 10_000 + shotIndex, rallyIndex: rallyIndex,
                             shotIndex: shotIndex, side: shot?.hit.hitter ?? rally.winner,
                             shotType: shot?.shotType ?? .unknown, point: landing,
                             isIn: isIn, marginM: margin, confidence: confidence)
        }
    }

    private static func makeNotices(_ analysis: MatchAnalysis, summary: MatchSummary) -> [ResultsNotice] {
        var notices: [ResultsNotice] = []
        if analysis.camera == nil {
            notices.append(ResultsNotice(id: "calibration", kind: .calibration, title: "球场尚未标定",
                                         message: "这场没有可靠的球场标定，因此不显示球速、落点和米制移动数据。"))
        }
        let lowRally = analysis.rallies.contains { $0.confidence > 0 && $0.confidence < 0.55 }
        let lowOutcome = analysis.rallies.contains { $0.outcomeConfidence > 0 && $0.outcomeConfidence < 0.55 }
        if lowRally || lowOutcome {
            notices.append(ResultsNotice(id: "confidence", kind: .confidence, title: "部分判定信心较低",
                                         message: "低信心的回合或落点已保留，但比分和结束方式可能需要你回看确认。"))
        }
        if analysis.rallies.isEmpty {
            notices.append(ResultsNotice(id: "empty", kind: .empty, title: "还没有识别到回合",
                                         message: "可以回看原视频，或确认画面包含完整球场和羽毛球。"))
        }
        _ = summary
        return notices
    }

    // This is the Python players.movement_stats algorithm, kept here so the UI's
    // precomputed numbers remain stable even when thousands of observations arrive.
    private static func movementStats(_ observations: [PlayerObservation], side: CourtSide, meta: VideoMeta) -> MovementStats {
        let groups = Dictionary(grouping: observations.filter {
            $0.side == side && $0.footWorld?.isFinite == true
        }, by: \.trackID)
        let per = groups.compactMapValues { statsOne($0.sorted { $0.frame < $1.frame }, meta: meta) }
        guard !per.isEmpty else { return MovementStats() }
        let tracked = per.values.reduce(0) { $0 + $1.trackedSeconds }
        let weightedHeat = zipHeatmaps(per.values.map { ($0.heatmap, $0.trackedSeconds) })
        let distance = per.values.reduce(0) { $0 + $1.distanceM }
        return MovementStats(
            distanceM: distance.rounded(toPlaces: 2),
            averageSpeedMS: tracked > 0 ? (distance / tracked).rounded(toPlaces: 3) : 0,
            maxSpeedMS: per.values.map(\.maxSpeedMS).max() ?? 0,
            heatmap: weightedHeat,
            courtCoverage: weightedCoverage(Array(per.values)),
            trackedSeconds: tracked.rounded(toPlaces: 2),
            playerStats: per
        )
    }

    private static func statsOne(_ observations: [PlayerObservation], meta: VideoMeta) -> PlayerMovementStats? {
        guard observations.count > 1 else { return nil }
        let points = observations.compactMap { observation -> (Double, SIMD2<Double>)? in
            guard let point = observation.footWorld, point.isFinite else { return nil }
            return (meta.time(ofFrame: observation.frame), point)
        }
        guard points.count > 1 else { return nil }
        var runs: [[(Double, SIMD2<Double>)]] = [[]]
        for point in points {
            if let last = runs.last?.last, point.0 - last.0 > 0.5 { runs.append([]) }
            runs[runs.count - 1].append(point)
        }
        var distance = 0.0
        var tracked = 0.0
        var speeds: [Double] = []
        var dwell = Array(repeating: Array(repeating: 0.0, count: 10), count: 12)
        for run in runs where run.count > 1 {
            let times = run.map(\.0)
            let positions = run.map(\.1)
            tracked += times.last! - times.first!
            let end = times.last!
            let start = times.first!
            var sample = start
            var resampled: [SIMD2<Double>] = []
            while sample <= end + 1e-9 {
                resampled.append(interpolate(sample, times: times, points: positions))
                sample += 0.1
            }
            if resampled.count > 1 {
                distance += zip(resampled, resampled.dropFirst()).reduce(0) {
                    $0 + simd_distance($1.0, $1.1)
                }
            }
            for i in positions.indices {
                let lowTime = times[i] - 0.15
                let highTime = times[i] + 0.15
                let lowFraction = fractionalIndex(lowTime, in: times)
                let highFraction = fractionalIndex(highTime, in: times)
                let low = max(0, min(times.count - 1, Int(floor(lowFraction))))
                let high = max(0, min(times.count - 1, Int(ceil(highFraction))))
                if times[high] - times[low] >= 0.15 {
                    let speed = simd_distance(positions[high], positions[low]) / (times[high] - times[low])
                    if speed <= 9 { speeds.append(speed) }
                }
                let dt: Double
                if i + 1 < times.count { dt = times[i + 1] - times[i] } else { dt = times[i] - times[max(0, i - 1)] }
                // Clamp in Double first: Int() traps on huge (but finite) positions.
                let r = (abs(positions[i].x) / CourtModel.halfLength * 12).clampedCell(11)
                let c = ((positions[i].y + CourtModel.doublesHalfWidth) / CourtModel.doublesWidth * 10).clampedCell(9)
                let row = r, col = c
                dwell[row][col] += max(0, dt)
            }
        }
        let heatTotal = dwell.flatMap { $0 }.reduce(0, +)
        let heat = heatTotal > 0 ? dwell.map { $0.map { $0 / heatTotal } } : dwell
        let maxSpeed = speeds.isEmpty ? 0 : percentile(speeds, 0.95)
        let coverage = Double(dwell.flatMap { $0 }.filter { $0 >= 0.2 }.count) / 120.0
        return PlayerMovementStats(distanceM: distance.rounded(toPlaces: 2),
                                   averageSpeedMS: tracked > 0 ? (distance / tracked).rounded(toPlaces: 3) : 0,
                                   maxSpeedMS: maxSpeed.rounded(toPlaces: 3), heatmap: heat,
                                   courtCoverage: coverage.rounded(toPlaces: 3),
                                   trackedSeconds: tracked.rounded(toPlaces: 2),
                                   dwellSeconds: dwell)
    }

    private static func interpolate(_ time: Double, times: [Double], points: [SIMD2<Double>]) -> SIMD2<Double> {
        if time <= times[0] { return points[0] }
        if time >= times[times.count - 1] { return points[points.count - 1] }
        var high = times.count - 1
        var low = 0
        while high - low > 1 {
            let middle = (low + high) / 2
            if times[middle] <= time { low = middle } else { high = middle }
        }
        let fraction = (time - times[low]) / (times[high] - times[low])
        return points[low] + (points[high] - points[low]) * fraction
    }

    private static func fractionalIndex(_ value: Double, in values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        if value <= values[0] { return 0 }
        if value >= values[values.count - 1] { return Double(values.count - 1) }
        var high = values.count - 1
        var low = 0
        while high - low > 1 {
            let middle = (low + high) / 2
            if values[middle] <= value { low = middle } else { high = middle }
        }
        let span = values[high] - values[low]
        return span == 0 ? Double(low) : Double(low) + (value - values[low]) / span
    }

    private static func percentile(_ values: [Double], _ q: Double) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        let position = q * Double(sorted.count - 1)
        let low = Int(position.rounded(.down))
        let high = min(low + 1, sorted.count - 1)
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
    }

    private static func zipHeatmaps(_ values: [([[Double]], Double)]) -> [[Double]] {
        let total = values.reduce(0) { $0 + $1.1 }
        guard total > 0 else { return Array(repeating: Array(repeating: 0, count: 10), count: 12) }
        return (0..<12).map { row in
            (0..<10).map { col in values.reduce(0) { $0 + $1.0[row][col] * $1.1 } / total }
        }
    }

    private static func weightedCoverage(_ values: [PlayerMovementStats]) -> Double {
        var dwell = Array(repeating: Array(repeating: 0.0, count: 10), count: 12)
        for value in values {
            for row in 0..<12 {
                for col in 0..<10 { dwell[row][col] += value.dwellSeconds[row][col] }
            }
        }
        let covered = dwell.flatMap { $0 }.filter { $0 >= 0.2 }.count
        return (Double(covered) / 120.0).rounded(toPlaces: 3)
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? { if case .object(let value) = self { return value }; return nil }
    var intValue: Int? {
        switch self {
        case .number(let value): return value.isFinite && abs(value) < 1e9 ? Int(value) : nil
        case .string(let value): return Int(value)
        default: return nil
        }
    }
    var boolValue: Bool? { if case .bool(let value) = self { return value }; return nil }
}

private extension Double {
    /// Python round(x, n) parity: format on the exact binary value (round-half-even at the
    /// decimal digit), which scaling by 10^n and rounding does not reproduce.
    func rounded(toPlaces places: Int) -> Double {
        guard isFinite else { return self }
        return Double(String(format: "%.*f", places, self)) ?? self
    }

    func clampedCell(_ last: Int) -> Int {
        guard isFinite else { return 0 }
        return Int(Swift.min(Swift.max(self, 0), Double(last)))
    }
}
