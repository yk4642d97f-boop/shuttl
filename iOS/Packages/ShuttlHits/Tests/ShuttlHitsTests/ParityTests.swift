import Foundation
import ShuttlCore
import XCTest
@testable import ShuttlHits

/// Swift port vs the Python reference (backend/scripts/swift_fixtures/hits.py).
final class ParityTests: XCTestCase {
    override func setUp() { HitModelRegistry.install(nil) }

    private func run(_ c: FCase, _ par: Parity, rescorer: (any HitRescorer)? = nil) -> [HitEvent] {
        let inp = c.inputs
        let hs = detectHits(track: inp.track, meta: inp.meta, camera: inp.camera, players: inp.players,
                            config: c.config, rescorer: rescorer)
        par.hits(hs, c.hits, c.name)
        if let pc = c.cands {
            let tr = c.config.useRawXY ? Track2D(t: inp.track.t, xy: Hits.preferRaw(inp.track), conf: inp.track.conf,
                                                 visible: inp.track.visible, interpolated: inp.track.interpolated)
                : inp.track
            par.cands(detectCandidates(track: tr, meta: inp.meta, camera: inp.camera, players: inp.players,
                                       config: c.config, rescorer: rescorer), pc, c.name)
        }
        return hs
    }

    func testUnitCases() throws {
        let f = try Fixtures.load("hits_unit.json")
        let par = Parity()
        for c in f.cases {
            _ = run(c, par)
            if let s = c.scan { checkScan(c, s, par) }
        }
        par.assertClean("unit")
    }

    /// Intermediates of detect_candidates on the cases that carry a scan dump.
    private func checkScan(_ c: FCase, _ s: FScan, _ par: Parity) {
        let inp = c.inputs
        let cfg = c.config
        let xy = cfg.useRawXY ? Hits.preferRaw(inp.track) : inp.track.xy
        var oi = Hits.observed(inp.track, xy: xy)
        var oxy = oi.map { xy[$0] }
        let cc = Hits.contentClock(frames: oi, xy: oxy, trackT: inp.track.t)
        var t: [Double] = []
        var o2: [Int] = [], x2: [SIMD2<Double>] = []
        for i in oi.indices where cc.keep[i] { o2.append(oi[i]); t.append(cc.t[i]); x2.append(oxy[i]) }
        oi = o2; oxy = x2
        let diag = NP.pyHypot(Double(inp.meta.width), Double(inp.meta.height))
        let sigma = cfg.sigmaPx ?? Hits.estimateSigma(t: t, xy: oxy)
        par.dev("scan.sigma", sigma, s.sigma, tol: 0, "\(c.name)")
        let keep = Hits.rejectOutliers(t: t, xy: oxy, sigma: sigma, diag: diag)
        let t2 = t.indices.filter { keep[$0] }.map { t[$0] }
        let f2 = oi.indices.filter { keep[$0] }.map { oi[$0] }
        let x3 = oxy.indices.filter { keep[$0] }.map { oxy[$0] }
        par.same("scan.obs_frames", f2, s.obs_frames, "\(c.name)")
        par.same("scan.obs_t", t2, s.obs_t, "\(c.name)")
        guard f2 == s.obs_frames else { return }
        let grid = inp.track.t.filter { $0 >= t2[0] && $0 <= t2[t2.count - 1] }
        let sc = Hits.scan(Fitter(t: t2, xy: x3), taus: grid, cfg: cfg, sigma: sigma)
        par.same("scan.ok", sc.ok, s.ok, "\(c.name)")
        par.same("scan.lo", sc.lo, s.lo, "\(c.name)")
        par.same("scan.hi", sc.hi, s.hi, "\(c.name)")
        par.same("scan.m", sc.m, s.m, "\(c.name)")
        for g in grid.indices where sc.ok[g] {
            // absolute scale: lam of a no-kink window is O(1) and pure round-off noise
            par.dev("scan.lam_v", sc.lamV[g], s.lam_v[g], tol: 1e-6 * Swift.max(1, abs(s.lam_v[g])), "\(c.name) g\(g)")
            par.dev("scan.lam", sc.lam[g], s.lam[g], tol: 1e-6 * Swift.max(1, abs(s.lam[g])), "\(c.name) g\(g)")
        }
    }

    func testSimE2E() throws {
        let f = try Fixtures.load("hits_sim_e2e.json")
        let par = Parity()
        var met = HitMetrics()
        var doubles = 0, singles = 0
        for c in f.cases {
            let hs = run(c, par)
            met.add(pred: hs, trueFrames: c.true_frame!, trueSides: c.true_side!)
            if c.doubles == true { doubles += 1 } else { singles += 1 }
        }
        par.assertClean("sim_e2e")
        print(String(format: "[sim_e2e] %d singles + %d doubles: tp %d fp %d fn %d  P %.3f R %.3f F1 %.4f hitter_acc %.4f",
                     singles, doubles, met.tp, met.nPred - met.tp, met.nTrue - met.tp, met.precision, met.recall,
                     met.f1, met.hitterAcc))
        XCTAssertGreaterThan(doubles, 0); XCTAssertGreaterThan(singles, 0)
        // Python hits.py on these exact inputs = the fixture's expected hits (hidden contacts on since
        // 2026-09-28 phase 2; was tp 105 / pred 121 / F1 0.857 in ml/benchmarks/20260927_115112_e2e.json):
        // tp 106 fp 17 fn 18, F1 0.8582995951417004, hitter 0.9622641509433962 (recomputed 2026-09-29, accuracy2.log)
        XCTAssertEqual(met.tp, 106); XCTAssertEqual(met.nPred, 123); XCTAssertEqual(met.nTrue, 124)
        XCTAssertEqual(met.f1, 0.8582995951417004, accuracy: 1e-12)
        XCTAssertEqual(met.hitterAcc, 0.9622641509433962, accuracy: 1e-12)
    }

    /// Low 'behind' camera, 5-rally matches: exercises the hidden contacts above the frame.
    func testBehindMultiRally() throws {
        let f = try Fixtures.load("hits_behind.json")
        let par = Parity()
        var met = HitMetrics()
        var nHidden = 0
        for c in f.cases {
            let hs = run(c, par)
            met.add(pred: hs, trueFrames: c.true_frame!, trueSides: c.true_side!)
            nHidden += (c.cands ?? []).filter { $0.hidden == true }.count
        }
        par.assertClean("behind")
        XCTAssertGreaterThan(nHidden, 0)
        print(String(format: "[behind] %d cases, %d hidden candidates: tp %d fp %d fn %d  F1 %.4f hitter_acc %.4f",
                     f.cases.count, nHidden, met.tp, met.nPred - met.tp, met.nTrue - met.tp, met.f1, met.hitterAcc))
    }

    func testScenarios() throws {
        let f = try Fixtures.load("hits_scen.json")
        let par = Parity()
        for c in f.cases { _ = run(c, par) }
        par.assertClean("scenarios")
    }

    /// F4 gap contact (hits.py _gap_contacts): synthetic PTS holes (test_hits.test_gap_contact_after_a_pts_gap)
    /// and the robustness F4 clip pair through pipeline_v2's stages. Parity per candidate + hit, and the
    /// behaviour: a hole ending at the contact gives one 'gap_contact' hit, a hole inside a flight none.
    func testGapContacts() throws {
        let f = try Fixtures.load("hits_gap.json")
        let par = Parity()
        var byName: [String: [HitEvent]] = [:]
        for c in f.cases { byName[c.name] = run(c, par) }
        par.assertClean("gap")
        XCTAssertEqual(f.cases.count, 10)
        for fps in [30, 60] {
            XCTAssertEqual(byName["gap_hidden_\(fps)"]?.map(\.source), ["gap_contact"])
            XCTAssertEqual(byName["gap_hidden_\(fps)"]?.first?.t ?? 0, 1.0, accuracy: 1e-9)
            XCTAssertEqual(byName["gap_hidden_off_\(fps)"]?.count, 0)
            XCTAssertEqual(byName["gap_mid_out_\(fps)"]?.map(\.source), ["kink"])
            XCTAssertEqual(byName["gap_mid_in_\(fps)"]?.map(\.source), ["kink"])
        }
        let drop = byName["f4_real_drop"] ?? []
        XCTAssertEqual(drop.map(\.frame), [50, 61, 100]); XCTAssertEqual(drop.map(\.hitter), [.near, .far, .near])
        XCTAssertEqual(drop.map(\.source), ["kink", "gap_contact", "kink"])
        XCTAssertTrue(drop[1].isInferred && !drop[0].isInferred)
        XCTAssertEqual(byName["f4_real_reencode"]?.map(\.frame), [50, 73, 112])
    }

    func testRealClips() throws {
        let f = try Fixtures.load("hits_real.json")
        let par = Parity()
        for c in f.cases { _ = run(c, par) }
        XCTAssertEqual(f.cases.count, 29)
        par.assertClean("real")
    }

    func testLearnedLinearModel() throws {
        let model = try LinearHitModel.load(contentsOf: Fixtures.url("hits_model.json"))
        XCTAssertEqual(model.w.count, HIT_FEATURES.count)
        XCTAssertEqual(model.features ?? [], HIT_FEATURES)
        let zero = LinearHitModel(w: [Double](repeating: 0, count: HIT_FEATURES.count), b: 0)
        let lf = try Fixtures.load("hits_learned.json", as: FLearned.self)
        var sources: [String: [String: FCase]] = [:]
        let par = Parity()
        for lc in lf.cases {
            if sources[lc.source] == nil {
                sources[lc.source] = Dictionary(uniqueKeysWithValues: try Fixtures.load(lc.source).cases.map { ($0.name, $0) })
            }
            let src = sources[lc.source]![lc.name]!
            let c = FCase(name: "\(lc.model):\(lc.name)", track: src.track, meta: src.meta, camera: src.camera,
                          players: src.players, config: src.config, hits: lc.hits, cands: lc.cands, scan: nil,
                          true_frame: nil, true_side: nil, doubles: nil)
            _ = run(c, par, rescorer: lc.model == "zero" ? zero : model)
        }
        par.assertClean("learned")
    }

    func testRegistryInstallMatchesExplicitRescorer() throws {
        let model = try LinearHitModel.load(contentsOf: Fixtures.url("hits_model.json"))
        let lf = try Fixtures.load("hits_learned.json", as: FLearned.self)
        let src = try Fixtures.load("hits_unit.json").cases.first { $0.name == "contract_arc" }!
        let inp = src.inputs
        XCTAssertTrue(HitModelRegistry.load(path: Fixtures.url("hits_model.json").path))
        defer { HitModelRegistry.install(nil) }
        let viaRegistry = detectHits(track: inp.track, meta: inp.meta)
        let explicit = detectHits(track: inp.track, meta: inp.meta, rescorer: model)
        XCTAssertEqual(viaRegistry, explicit)
        let par = Parity()
        par.hits(viaRegistry, lf.cases.first { $0.model == "synthetic" && $0.name == "contract_arc" }!.hits, "registry")
        par.assertClean("registry")
        XCTAssertFalse(HitModelRegistry.load(path: "/nonexistent/missing.json"))
        XCTAssertNil(HitModelRegistry.installed)
    }
}
