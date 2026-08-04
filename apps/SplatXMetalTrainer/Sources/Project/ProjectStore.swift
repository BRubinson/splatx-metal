import Foundation
import Observation

/// The single source of truth for the projects root directory. The folder name is authoritative as
/// the project name, and the manifest (project.json) is metadata. Creation assembles into a
/// ".tmp-*" folder and then renames atomically — a failed import leaves no ghost project on the
/// Home screen.
@MainActor @Observable final class ProjectStore {
    static let manifestName = "project.json"

    let rootURL: URL
    private(set) var projects: [Project] = []
    private(set) var scanWarnings: [String] = []

    init(rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("SplatXMetalTrainer/Projects", isDirectory: true)) {
        self.rootURL = rootURL
    }

    func directory(for project: Project) -> URL {
        rootURL.appendingPathComponent(project.name, isDirectory: true)
    }

    func scan() {
        projects = []; scanWarnings = []
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: rootURL, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for dir in entries {
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true,
                  !dir.lastPathComponent.hasPrefix(".") else { continue }
            do {
                let data = try Data(contentsOf: dir.appendingPathComponent(Self.manifestName))
                var p = try JSONDecoder.project.decode(Project.self, from: data)
                p.name = dir.lastPathComponent   // the folder name is authoritative (picking up a manual rename)
                projects.append(p)
            } catch {
                scanWarnings.append("skipped \(dir.lastPathComponent): \(error.localizedDescription)")
            }
        }
        projects.sort { $0.createdAt > $1.createdAt }
    }

    /// Once assemble has filled the temporary directory with content, it writes the manifest and
    /// atomically moves it to the final name.
    /// The return value of assemble is the imageCount.
    func create(name rawName: String, sourceKind: ProjectSourceKind, state: ProjectState,
                assemble: (URL) throws -> Int) throws -> Project {
        let fm = FileManager.default
        try fm.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let tmp = rootURL.appendingPathComponent(".tmp-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        do {
            let imageCount = try assemble(tmp)
            let name = uniqueName(from: rawName)
            let project = Project(id: UUID(), name: name, sourceKind: sourceKind, state: state,
                                  imageCount: imageCount, alignSummary: nil, runs: [], createdAt: Date())
            try writeManifest(project, into: tmp)
            try fm.moveItem(at: tmp, to: rootURL.appendingPathComponent(name, isDirectory: true))
            projects.insert(project, at: 0)
            return project
        } catch {
            try? fm.removeItem(at: tmp)
            throw error
        }
    }

    func save(_ project: Project) throws {
        try writeManifest(project, into: directory(for: project))
        if let i = projects.firstIndex(where: { $0.id == project.id }) { projects[i] = project }
    }

    func delete(_ project: Project) throws {
        try FileManager.default.removeItem(at: directory(for: project))
        projects.removeAll { $0.id == project.id }
    }

    /// Moving the folder is the authoritative operation of a rename — **it throws only on a
    /// moveItem failure** (the caller's invariant: if rename throws, the folder did not move). A
    /// manifest write failure after a successful move is non-fatal — the folder name is
    /// authoritative and scan() overwrites the manifest name with the folder name, so the next scan
    /// self-heals. The in-memory projects update is always performed so that the return value, the
    /// list, and the file system all point consistently at the new name.
    func rename(_ project: Project, to rawName: String) throws -> Project {
        let src = directory(for: project)
        var renamed = project
        renamed.name = uniqueName(from: rawName)
        try FileManager.default.moveItem(at: src, to: directory(for: renamed))
        try? writeManifest(renamed, into: directory(for: renamed))
        if let i = projects.firstIndex(where: { $0.id == renamed.id }) { projects[i] = renamed }
        return renamed
    }

    private func uniqueName(from raw: String) -> String {
        var base = raw.replacingOccurrences(of: "/", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if base.isEmpty { base = "Project" }
        var name = base
        var n = 2
        while FileManager.default.fileExists(atPath: rootURL.appendingPathComponent(name).path) {
            name = "\(base)-\(n)"; n += 1
        }
        return name
    }

    private func writeManifest(_ project: Project, into dir: URL) throws {
        let data = try JSONEncoder.project.encode(project)
        try data.write(to: dir.appendingPathComponent(Self.manifestName), options: .atomic)
    }
}
