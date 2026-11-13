import SwiftUI
import SwiftData

@main
struct ShuttlApp: App {
    @UIApplicationDelegateAdaptor(ShuttlAppDelegate.self) private var appDelegate
    let modelContainer: ModelContainer
    /// App-lifetime upload/analysis coordinator (outlives the recording screen).
    @State private var pipeline: MatchPipeline
    @State private var authService = AuthService()

    init() {
        DisplayFace.register()
        let container = Self.makeContainer()
        modelContainer = container
        let pipeline = MatchPipeline.live(container: container)
        #if DEBUG
        if DemoMode.isOn { pipeline.seedDemoActivity() }
        #endif
        _pipeline = State(initialValue: pipeline)
    }

    private static func makeContainer() -> ModelContainer {
        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        #if DEBUG
        if DemoMode.isOn { return DemoMode.makeContainer(schema: schema) }
        #endif
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("SwiftData 容器初始化失败: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(authService)
                .environment(pipeline)
        }
        .modelContainer(modelContainer)
    }
}
