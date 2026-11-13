import XCTest

/// Drives the real app against the local backend (uvicorn on 127.0.0.1:8000):
/// register → land on the tab bar → visit every tab. Screenshots are attached
/// to the test result.
final class ShuttlUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testRegisterThenBrowseAllTabs() throws {
        let app = XCUIApplication()
        app.launchArguments += ["-UITestResetAuth"]

        // The record tab asks for camera access once the tabs appear.
        addUIInterruptionMonitor(withDescription: "Camera permission") { alert in
            for label in ["允许", "允許", "Allow", "OK", "好"] where alert.buttons[label].exists {
                alert.buttons[label].tap()
                return true
            }
            return false
        }

        app.launch()
        attachScreenshot(named: "01-login")

        // Switch to register mode.
        let toRegister = app.buttons["还没有账号？注册"]
        XCTAssertTrue(toRegister.waitForExistence(timeout: 10), "登录页未出现")
        toRegister.tap()

        let email = "uitest_\(Int(Date().timeIntervalSince1970))@shuttl.app"
        fill(app.textFields["邮箱地址"], with: email)
        fill(app.secureTextFields["密码"], with: "testpass123")
        fill(app.textFields["你的球场名号"], with: "uitest")
        attachScreenshot(named: "02-register-filled")

        app.buttons["注册"].tap()

        // A successful round-trip through APIClient → FastAPI flips the root to the tab bar.
        let matchesTab = app.buttons["比赛"]
        XCTAssertTrue(matchesTab.waitForExistence(timeout: 20),
                      "注册后没有进入主界面——检查后端是否在 127.0.0.1:8000 运行")

        // Poke the app so a pending permission alert gets routed to the monitor.
        app.tap()
        attachScreenshot(named: "03-record-tab")

        matchesTab.tap()
        attachScreenshot(named: "04-matches-tab")

        app.buttons["统计"].tap()
        attachScreenshot(named: "05-statistics-tab")
    }

    // MARK: - Helpers

    private func fill(_ field: XCUIElement, with text: String) {
        XCTAssertTrue(field.waitForExistence(timeout: 5), "找不到输入框")
        field.tap()
        field.typeText(text)
    }

    private func attachScreenshot(named name: String) {
        // Give SwiftUI transitions a moment to settle.
        Thread.sleep(forTimeInterval: 0.8)
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }
}
