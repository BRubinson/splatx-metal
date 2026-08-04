import SwiftUI

/// Fuzzy ranking. It scores command palette candidates by case-insensitive subsequence matching.
///
/// - A `nil` from `rank` means no match (the query characters do not appear in order).
/// - **Lower scores are better**: the earlier the first match (a prefix) and the more contiguous
///   the matches, the lower the score.
///   A greedy left-to-right scan — sufficient for palette ordering, and deterministic.
enum Fuzzy {
    /// A score when `query` is a (case-insensitive) subsequence of `candidate`, otherwise `nil`.
    /// An empty query matches everything (score 0, tied at the top).
    static func rank(query: String, candidate: String) -> Int? {
        let q = Array(query.lowercased())
        if q.isEmpty { return 0 }
        let c = Array(candidate.lowercased())

        var score = 0
        var prev = -1        // the index of the previous match (-1 = none yet)
        var qi = 0           // the next query character to match
        var ci = 0
        while qi < q.count && ci < c.count {
            if c[ci] == q[qi] {
                // First match: a start-position penalty (the later it starts, the worse).
                // Later matches: a gap penalty against the previous one (the more scattered, the
                // worse; 0 when contiguous).
                score += prev < 0 ? ci : (ci - prev - 1)
                prev = ci
                qi += 1
            }
            ci += 1
        }
        // If the query is not consumed to the end, it is not a subsequence.
        return qi == q.count ? score : nil
    }
}

/// The ⌘K command palette. It filters and sorts the enabled commands of `CommandRegistry` (the same
/// single source of truth as the menu bar and the shortcuts) with `Fuzzy.rank` — it does not create
/// a second command source.
///
/// Keys: typing = filter, ↑/↓ = move the selection, Enter = execute and close, Esc or a tap on the
/// background = close.
struct CommandPalette: View {
    let registry: CommandRegistry
    @Binding var isPresented: Bool

    @State private var query = ""
    @State private var selection = 0
    @FocusState private var fieldFocused: Bool

    /// Stably sorts the enabled commands by (score, registration order). The command that opens the
    /// palette itself is excluded.
    private var results: [AppCommand] {
        registry.enabled()
            .filter { $0.id != AppCommandID.openPalette }
            .compactMap { cmd in Fuzzy.rank(query: query, candidate: cmd.title).map { (cmd, $0) } }
            .enumerated()
            .sorted { ($0.element.1, $0.offset) < ($1.element.1, $1.offset) }
            .map { $0.element.0 }
    }

    var body: some View {
        ZStack(alignment: .top) {
            // The background dim — tapping it closes the palette.
            Color.black.opacity(0.4)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { close() }

            palette
                .frame(maxWidth: 560)
                .padding(.top, 96)
                .padding(.horizontal, 24)
        }
        .onExitCommand { close() }          // Esc
        .onChange(of: query) { selection = 0 }   // every keystroke returns the selection to the top
        .onAppear {
            query = ""; selection = 0
            fieldFocused = true
        }
    }

    private var palette: some View {
        VStack(spacing: 0) {
            // The search field
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(SplatXTheme.textSecondary)
                TextField("Run a command…", text: $query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 16))
                    .foregroundStyle(SplatXTheme.textPrimary)
                    .focused($fieldFocused)
                    .onSubmit { runSelected() }       // Enter
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)

            Divider().overlay(SplatXTheme.hairline)

            // The results list
            if results.isEmpty {
                Text("No commands")
                    .font(.callout)
                    .foregroundStyle(SplatXTheme.textSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14).padding(.vertical, 14)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(Array(results.enumerated()), id: \.element.id) { index, cmd in
                                row(cmd, selected: index == selection)
                                    .id(index)
                                    .contentShape(Rectangle())
                                    .onTapGesture { selection = index; runSelected() }
                            }
                        }
                    }
                    .frame(maxHeight: 320)
                    .onChange(of: selection) { _, new in
                        proxy.scrollTo(new, anchor: .center)
                    }
                }
            }
        }
        .background(SplatXTheme.elevated)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(SplatXTheme.hairline))
        .shadow(color: .black.opacity(0.45), radius: 24, y: 12)
        // ↑/↓ move the selection (caught at the container even when the field has focus).
        .onKeyPress(.upArrow) { move(-1); return .handled }
        .onKeyPress(.downArrow) { move(1); return .handled }
    }

    private func row(_ cmd: AppCommand, selected: Bool) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text(cmd.title)
                    .font(.system(size: 14))
                    .foregroundStyle(SplatXTheme.textPrimary)
                if let sub = cmd.subtitle {
                    Text(sub)
                        .font(.system(size: 11))
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
            }
            Spacer(minLength: 8)
            if let label = Self.shortcutLabel(cmd.shortcut) {
                Text(label)
                    .font(.system(size: 12, weight: .medium).monospaced())
                    .foregroundStyle(SplatXTheme.textSecondary)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 9)
        .background(selected ? SplatXTheme.accentDim : .clear)
    }

    // MARK: - Actions

    private func move(_ delta: Int) {
        let count = results.count
        guard count > 0 else { return }
        selection = min(max(selection + delta, 0), count - 1)
    }

    private func runSelected() {
        let r = results
        guard r.indices.contains(selection) else { return }
        registry.execute(id: r[selection].id)
        close()
    }

    private func close() {
        isPresented = false
    }

    /// KeyboardShortcut → the display symbols (⌃⌥⇧⌘ + the key). `nil` when there is no shortcut.
    static func shortcutLabel(_ sc: KeyboardShortcut?) -> String? {
        guard let sc else { return nil }
        var s = ""
        if sc.modifiers.contains(.control) { s += "⌃" }
        if sc.modifiers.contains(.option)  { s += "⌥" }
        if sc.modifiers.contains(.shift)   { s += "⇧" }
        if sc.modifiers.contains(.command) { s += "⌘" }
        s += String(sc.key.character).uppercased()
        return s
    }
}
