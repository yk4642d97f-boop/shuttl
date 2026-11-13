import SwiftUI

// MARK: - LoginView
// A poster, not a form: Bauhaus-style shapes bleed off the top edge, a huge
// serif wordmark anchors the left, and the form sits below on paper.

struct LoginView: View {
    @Environment(AuthService.self) private var auth
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var email      = ""
    @State private var password   = ""
    @State private var nickname   = ""
    @State private var isRegister = false
    @State private var isLoading  = false
    @State private var errorMessage: String?
    @State private var appeared = false

    // Track whether a field has been edited at least once (to gate inline errors)
    @State private var emailTouched    = false
    @State private var passwordTouched = false

    @FocusState private var focusedField: Field?

    enum Field: Hashable { case email, password, nickname }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                PosterArt()
                    .frame(height: 300)
                    .frame(maxWidth: .infinity)
                    .stagger(0, appeared: appeared)

                headerSection
                    .padding(.horizontal, AppSpacing.gutter)
                    .padding(.top, -14)
                    .padding(.bottom, AppSpacing.xl)
                    .stagger(1, appeared: appeared)

                formSection
                    .padding(.horizontal, AppSpacing.gutter)
                    .stagger(2, appeared: appeared)

                if let msg = errorMessage {
                    errorBanner(msg)
                        .padding(.horizontal, AppSpacing.gutter)
                        .padding(.top, AppSpacing.md)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                actionSection
                    .padding(.horizontal, AppSpacing.gutter)
                    .padding(.top, AppSpacing.lg)
                    .padding(.bottom, AppSpacing.xxl)
                    .stagger(3, appeared: appeared)
            }
        }
        .scrollDismissesKeyboard(.interactively)
        .scrollBounceBehavior(.basedOnSize)
        .ignoresSafeArea(edges: .top)
        .paperPage()
        .animation(AppAnimation.spring, value: isRegister)
        .animation(AppAnimation.standard, value: errorMessage)
        .onAppear {
            #if DEBUG
            if DemoMode.showsLoginErrors {
                isRegister = true
                email = "player.shuttl.app"
                password = "1234"
                emailTouched = true
                passwordTouched = true
                errorMessage = "该邮箱已被注册，换一个试试？"
            }
            #endif
            if reduceMotion { appeared = true } else {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) { appeared = true }
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: AppSpacing.sm) {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                Text("Shuttl")
                    .font(AppFont.brand)
                    .foregroundStyle(Color.ink)
                Text(".")
                    .font(AppFont.brand)
                    .foregroundStyle(Color.tomato)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Shuttl")
            .accessibilityAddTraits(.isHeader)

            HStack(spacing: AppSpacing.sm) {
                Text(isRegister ? "办一张球场通行证" : "欢迎回到球场")
                    .font(AppFont.title3)
                    .foregroundStyle(Color.ink)
                    .contentTransition(.opacity)
                Sticker("AI 鹰眼", symbol: "eye.fill", fill: .pistachio, tilt: 4)
            }

            Text("每一拍的轨迹、球速与落点")
                .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
        }
    }

    // MARK: - Form

    private var formSection: some View {
        VStack(spacing: AppSpacing.md) {
            AuthField(
                label: "邮箱",
                placeholder: "邮箱地址",
                text: $email,
                keyboardType: .emailAddress,
                contentType: .emailAddress,
                isFocused: focusedField == .email,
                error: emailTouched ? emailError : nil
            )
            .focused($focusedField, equals: .email)
            .submitLabel(.next)
            .onSubmit { focusedField = .password }
            .onChange(of: focusedField) { old, _ in
                if old == .email { emailTouched = true }
            }

            AuthField(
                label: "密码",
                placeholder: "密码",
                text: $password,
                isSecure: true,
                contentType: isRegister ? .newPassword : .password,
                isFocused: focusedField == .password,
                error: passwordTouched ? passwordError : nil
            )
            .focused($focusedField, equals: .password)
            .submitLabel(isRegister ? .next : .go)
            .onSubmit { if isRegister { focusedField = .nickname } else { submit() } }
            .onChange(of: focusedField) { old, _ in
                if old == .password { passwordTouched = true }
            }

            if isRegister {
                AuthField(
                    label: "球场名号",
                    placeholder: "你的球场名号",
                    text: $nickname,
                    contentType: .nickname,
                    isFocused: focusedField == .nickname,
                    error: nil
                )
                .focused($focusedField, equals: .nickname)
                .submitLabel(.go)
                .onSubmit(submit)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }

    // MARK: - Error banner

    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: AppSpacing.sm) {
            Image(systemName: "exclamationmark")
                .font(.system(size: 12, weight: .black))
                .foregroundStyle(Color.tomato)
                .frame(width: 22, height: 22)
                .background(Color.ink, in: Circle())
            Text(message)
                .font(AppFont.callout.weight(.medium))
                .foregroundStyle(Color.ink)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(AppSpacing.md - 4)
        .background(Color.tomato, in: RoundedRectangle(cornerRadius: AppRadius.sm + 4, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: AppRadius.sm + 4, style: .continuous)
            .strokeBorder(Color.ink, lineWidth: 1.5))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("错误：\(message)")
    }

    // MARK: - Actions

    private var actionSection: some View {
        VStack(spacing: AppSpacing.md) {
            PillButton(
                title: isRegister ? "注册" : "登录",
                icon: isRegister ? "sparkles" : "arrow.right",
                isLoading: isLoading,
                style: .primary,
                action: submit
            )

            Button {
                withAnimation(AppAnimation.spring) {
                    isRegister.toggle()
                    errorMessage    = nil
                    emailTouched    = false
                    passwordTouched = false
                }
            } label: {
                (Text(isRegister ? "已有账号？" : "还没有账号？")
                    .foregroundColor(.inkSoft)
                 + Text(isRegister ? "登录" : "注册")
                    .foregroundColor(.cobalt)
                    .underline(true, color: .cobalt))
                    .font(AppFont.callout.weight(.semibold))
                    .frame(minHeight: 44)
            }
            .buttonStyle(ScaleButtonStyle())
            .accessibilityLabel(isRegister ? "已有账号？登录" : "还没有账号？注册")
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Validation

    private var emailError: String? {
        guard !email.isEmpty else { return nil }
        return email.contains("@") ? nil : "请输入有效的邮箱地址"
    }

    private var passwordError: String? {
        guard !password.isEmpty else { return nil }
        if isRegister { return password.count >= 8 ? nil : "密码不能少于 8 位" }
        return nil
    }

    private var formIsValid: Bool {
        let emailOK    = email.contains("@")
        let passwordOK = isRegister ? password.count >= 8 : !password.isEmpty
        return emailOK && passwordOK
    }

    // MARK: - Submit

    private func submit() {
        emailTouched    = true
        passwordTouched = true
        guard formIsValid else { return }

        focusedField = nil
        errorMessage = nil
        isLoading    = true

        Task { @MainActor in
            defer { isLoading = false }
            do {
                if isRegister {
                    try await auth.register(
                        email: email,
                        password: password,
                        nickname: nickname.isEmpty ? "球友" : nickname
                    )
                } else {
                    try await auth.login(email: email, password: password)
                }
            } catch {
                withAnimation(AppAnimation.standard) {
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - AuthField
// Mono label above a card-stock well. Focus draws a thicker cobalt rule;
// errors switch the rule to tomato and print the message underneath.

private struct AuthField: View {
    let label: String
    let placeholder: String
    @Binding var text: String
    var keyboardType: UIKeyboardType = .default
    var isSecure: Bool = false
    var contentType: UITextContentType? = nil
    var isFocused: Bool
    var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.xs + 2) {
            Text(label)
                .instrumentLabel(AppFont.labelSmall, color: isFocused ? .cobalt : .inkSoft)
                .accessibilityHidden(true)

            Group {
                if isSecure {
                    SecureField("", text: $text, prompt: Text(placeholder).foregroundColor(.inkFaint))
                } else {
                    TextField("", text: $text, prompt: Text(placeholder).foregroundColor(.inkFaint))
                        .keyboardType(keyboardType)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
            }
            .textContentType(contentType)
            .font(AppFont.body.weight(.medium))
            .foregroundStyle(Color.ink)
            .tint(.cobalt)
            .padding(.horizontal, AppSpacing.md)
            .frame(height: 54)
            .background(Color.card, in: RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: isFocused || error != nil ? 2 : 1.25)
            )
            .animation(AppAnimation.standard, value: isFocused)
            .accessibilityLabel(placeholder)
            .accessibilityIdentifier(placeholder)

            if let error {
                Label(error, systemImage: "exclamationmark.circle.fill")
                    .font(AppFont.caption.weight(.semibold))
                    .foregroundStyle(Color.tomatoText)
                    .padding(.horizontal, AppSpacing.xs)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(AppAnimation.standard, value: error)
    }

    private var borderColor: Color {
        if error != nil { return .tomato }
        if isFocused    { return .cobalt }
        return Color.ink.opacity(0.35)
    }
}

// MARK: - PosterArt
// Bauhaus composition that bleeds off the top: cobalt field, a butter sun,
// a lilac disc, a pistachio court strip, and the shuttlecock mid-flight.

private struct PosterArt: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var drift = false

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            ZStack(alignment: .topLeading) {
                UnevenRoundedRectangle(bottomLeadingRadius: 0, bottomTrailingRadius: 120, style: .continuous)
                    .fill(Color.cobalt)
                    .frame(width: w * 0.78, height: h)

                Halftone(color: .paper.opacity(0.35), spacing: 10, maxDot: 3.6, focus: .bottomTrailing)
                    .frame(width: w * 0.78, height: h)
                    .clipShape(UnevenRoundedRectangle(bottomTrailingRadius: 120, style: .continuous))

                Circle()
                    .fill(Color.butter)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: w * 0.62)
                    .position(x: w * 0.84, y: h * 0.36)

                Circle()
                    .fill(Color.lilac)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: w * 0.2)
                    .position(x: w * 0.14, y: h * 0.72)

                ZStack {
                    Color.pistachio
                    CourtPlan(.landscape).stroke(Color.card, lineWidth: 1.4)
                    CourtNet(.landscape).stroke(Color.ink, style: StrokeStyle(lineWidth: 2.4, dash: [2, 2]))
                }
                .frame(width: w * 0.56, height: w * 0.56 / CourtGeometry2D.aspect(.landscape))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.ink, lineWidth: 1.5))
                .rotation3DEffect(.degrees(58), axis: (x: 1, y: 0, z: 0), perspective: 0.6)
                .rotationEffect(.degrees(-8))
                .position(x: w * 0.72, y: h * 0.86)

                flightPath(w: w, h: h)

                ShuttlecockMark(skirt: .card, cork: .tomato, ink: .ink)
                    .frame(width: 104, height: 104)
                    .rotationEffect(.degrees(-128))
                    .position(x: w * 0.6, y: h * 0.42)
                    .offset(y: drift ? -6 : 4)
            }
        }
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 2.6).repeatForever(autoreverses: true)) { drift = true }
        }
        .accessibilityHidden(true)
    }

    /// Dotted arc the shuttle has just travelled.
    private func flightPath(w: CGFloat, h: CGFloat) -> some View {
        Path { p in
            p.move(to: CGPoint(x: w * 0.02, y: h * 0.9))
            p.addQuadCurve(to: CGPoint(x: w * 0.52, y: h * 0.5), control: CGPoint(x: w * 0.18, y: h * 0.3))
        }
        .stroke(Color.paper, style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [0.1, 11]))
    }
}

// MARK: - Staggered entrance

private extension View {
    func stagger(_ index: Int, appeared: Bool) -> some View {
        self
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 18)
            .animation(.spring(response: 0.6, dampingFraction: 0.8).delay(Double(index) * 0.07), value: appeared)
    }
}

// MARK: - Preview

#Preview("LoginView") {
    LoginView()
        .environment(AuthService())
}
