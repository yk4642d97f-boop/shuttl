import SwiftUI
import SwiftData

// Root: sign-in gate, then three sections under a floating ink tab bar.
// Each section owns one ink colour (record = tomato, matches = cobalt,
// statistics = pistachio) so you always know where you are by colour alone.

struct ContentView: View {
    @State private var chrome: AppChrome = {
        #if DEBUG
        if let tab = DemoMode.initialTab { return AppChrome(selectedTab: tab) }
        #endif
        if let tab = AppTab.launchArgument { return AppChrome(selectedTab: tab) }
        return AppChrome()
    }()
    /// The first-launch tab has been decided (an analysing/waiting match opens 比赛, once).
    @State private var pickedLaunchTab = false
    @Environment(AuthService.self) private var auth
    @Environment(MatchPipeline.self) private var pipeline
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.modelContext) private var modelContext

    /// On-device analysis needs no account (nothing is uploaded), so sign-in only
    /// gates the app when matches go to the server.
    private var canUseApp: Bool { auth.isLoggedIn || pipeline.analyzesOnDevice }

    var body: some View {
        // Swap the root instead of covering the tabs, so the camera tab (and its
        // permission prompt) only comes alive after sign-in.
        Group {
            if canUseApp {
                mainTabs
                    .transition(.opacity)
            } else {
                LoginView()
                    .transition(.opacity)
            }
        }
        .animation(AppAnimation.standard, value: canUseApp)
        .appTheme()
        // Pick up uploads/analyses left behind (app killed, suspended mid-upload, just signed in).
        .task(id: canUseApp) {
            guard canUseApp else { pipeline.suspendCloudWork(); return }
            pipeline.resumePending()
            openMatchesIfAnalysing()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active && canUseApp { pipeline.resumePending() }
        }
        .onReceive(NotificationCenter.default.publisher(for: .shuttlSessionExpired)) { _ in
            auth.refreshSessionState()
        }
        // Only the camera and replay rotate; leaving them snaps back to portrait.
        .onChange(of: wantsLandscape, initial: true) { _, allowed in
            OrientationPolicy.allowLandscape(allowed)
            #if DEBUG
            if allowed && DemoMode.forcesLandscape { OrientationPolicy.requestLandscape() }
            #endif
        }
    }

    /// Only replay really rotates. The camera stays portrait and counter-rotates its
    /// chrome (like Camera.app), so the preview never jumps mid-setup.
    private var wantsLandscape: Bool { canUseApp && (chrome.replayOpen || chrome.calibrationOpen) }

    /// Full-bleed moments hide the tab bar: replay, and the camera held on its side
    /// (rotate back to upright to switch sections).
    private var showsTabBar: Bool {
        !chrome.tabBarHidden && !chrome.replayOpen
            && !(chrome.selectedTab == .record && chrome.deviceLandscape)
    }

    /// Opening on 录像 would start the camera, which heats the phone and holds the analysis
    /// back. So when a match is analysing or waiting the app opens on 比赛 instead (once per
    /// launch; 录像 stays one tap away). A launch argument always wins.
    private func openMatchesIfAnalysing() {
        guard !pickedLaunchTab else { return }
        pickedLaunchTab = true
        #if DEBUG
        // -UIDemoLaunchCheck: start on the demo tab, then apply the real launch rule.
        if DemoMode.initialTab != nil && !ProcessInfo.processInfo.arguments.contains("-UIDemoLaunchCheck") { return }
        #endif
        guard AppTab.launchArgument == nil, chrome.selectedTab == .record else { return }
        // The stored status, not the live activity: at launch the pipeline tasks haven't reported yet.
        let stored = (try? modelContext.fetch(FetchDescriptor<Match>())) ?? []
        if pipeline.hasLocalWork || stored.contains(where: { $0.status == .analyzing }) {
            chrome.selectedTab = .matches
        }
    }

    private var mainTabs: some View {
        @Bindable var bindableChrome = chrome
        return Group {
            switch chrome.selectedTab {
            case .record:     RecordingView()
            case .matches:    MatchListView()
            case .statistics: StatisticsView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .environment(chrome)
        // An inset (not an overlay) so scroll content and the camera controls
        // stop above the bar instead of sliding underneath it.
        // Positions the floating bar. The inset does not reach scroll views inside the tabs'
        // NavigationStacks, so each of those adds `tabBarClearance()` itself.
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if showsTabBar {
                ShuttlTabBar(selectedTab: $bindableChrome.selectedTab)
                    .padding(.horizontal, AppSpacing.xl)
                    .padding(.bottom, AppSpacing.xs)
                    // A short 28 pt fade above the bar's top edge, then solid paper down to the
                    // screen edge: content never looks washed out, it just ends at the bar.
                    .background {
                        if chrome.selectedTab != .record {
                            VStack(spacing: 0) {
                                LinearGradient(colors: [Color.paper.opacity(0), Color.paper],
                                               startPoint: .top, endPoint: .bottom)
                                    .frame(height: 28)
                                Color.paper
                            }
                            .padding(.top, -28)
                            .padding(.horizontal, -AppSpacing.xl)
                            .ignoresSafeArea(edges: .bottom)
                            .allowsHitTesting(false)
                        }
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(AppAnimation.spring, value: showsTabBar)
    }
}

// MARK: - Tab bar clearance

extension View {
    /// Bottom content margin for a scroll view under the floating tab bar: the bar's height
    /// plus 16 pt, so the last card ends clear of the bar and its fade.
    func tabBarClearance() -> some View {
        contentMargins(.bottom, ShuttlTabBar.height + AppSpacing.md + AppSpacing.lg, for: .scrollContent)
    }
}

// MARK: - Chrome state shared with pushed screens

@MainActor
@Observable
final class AppChrome {
    /// Screens that need the whole canvas hide the tab bar while they're on screen.
    var tabBarHidden = false
    /// The replay screen is showing (full-bleed, may rotate).
    var replayOpen = false
    /// The camera screen sees the phone held on its side (gravity, not interface orientation).
    var deviceLandscape = false
    /// The corner-drag calibration screen is up (it may rotate: a landscape frame fills it better).
    var calibrationOpen = false
    /// Current section; screens switch it directly (e.g. "查看进度" after recording).
    var selectedTab: AppTab

    init(selectedTab: AppTab = .record) {
        self.selectedTab = selectedTab
    }
}

// MARK: - Tab definition

enum AppTab: CaseIterable {
    case record, matches, statistics

    /// `-ShuttlOpenTab record|matches|statistics` (device test scripts; release builds too).
    static var launchArgument: AppTab? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-ShuttlOpenTab"), i + 1 < args.count else { return nil }
        switch args[i + 1] {
        case "record": return .record
        case "matches": return .matches
        case "statistics": return .statistics
        default: return nil
        }
    }

    var icon: String {
        switch self {
        case .record:     return "record.circle.fill"
        case .matches:    return "rectangle.stack.fill"
        case .statistics: return "chart.pie.fill"
        }
    }

    var label: String {
        switch self {
        case .record:     return "录制"
        case .matches:    return "比赛"
        case .statistics: return "统计"
        }
    }

    /// The section's ink.
    var ink: Color {
        switch self {
        case .record:     return .tomato
        case .matches:    return .cobalt
        case .statistics: return .pistachio
        }
    }

    var onInk: Color {
        switch self {
        case .record:     return .onTomato
        case .matches:    return .onCobalt
        case .statistics: return .onPistachio
        }
    }
}

// MARK: - Floating ink tab bar

struct ShuttlTabBar: View {
    /// Capsule height: 48 pt items plus 4 pt padding top and bottom.
    static let height: CGFloat = 56
    @Binding var selectedTab: AppTab
    @Namespace private var blob

    var body: some View {
        HStack(spacing: AppSpacing.xxs) {
            ForEach(AppTab.allCases, id: \.self) { tab in
                TabBarItem(tab: tab, isSelected: selectedTab == tab, blob: blob) {
                    guard selectedTab != tab else { return }
                    withAnimation(AppAnimation.bouncy) { selectedTab = tab }
                }
            }
        }
        .padding(AppSpacing.xs)
        .background(Color.ink, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.paper.opacity(0.14), lineWidth: 1))
        .shadow(color: Color.ink.opacity(0.28), radius: 18, x: 0, y: 10)
        .sensoryFeedback(.selection, trigger: selectedTab)
    }
}

struct TabBarItem: View {
    let tab: AppTab
    let isSelected: Bool
    let blob: Namespace.ID
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: tab.icon)
                    .font(.system(size: 17, weight: .bold))
                    .symbolEffect(.bounce, value: isSelected)
                if isSelected {
                    Text(tab.label)
                        .font(DisplayFace.font(19, relativeTo: .subheadline))
                        .fixedSize()
                        .transition(.opacity.combined(with: .scale(scale: 0.6, anchor: .leading)))
                }
            }
            .foregroundStyle(isSelected ? tab.onInk : Color.paper.opacity(0.62))
            .frame(maxWidth: .infinity, minHeight: 48)
            .background {
                if isSelected {
                    Capsule()
                        .fill(tab.ink)
                        .matchedGeometryEffect(id: "blob", in: blob)
                }
            }
            .contentShape(Capsule())
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(tab.label)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}
