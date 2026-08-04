import XCTest
@testable import SplatXMetalTrainer

/// The fuzzy ranking contract: case-insensitive subsequence matching (no match = nil), and a
/// lower score is better.
/// A prefix or contiguous match scores lower than a scattered one (a bonus). This underpins the
/// ordering of the ⌘K palette.
final class FuzzyRankTests: XCTestCase {
    func testSubsequenceMatchAndMiss() {
        XCTAssertNotNil(Fuzzy.rank(query: "oc", candidate: "Open COLMAP"))   // the o..c subsequence
        XCTAssertNil(Fuzzy.rank(query: "zzz", candidate: "Open COLMAP"))
        XCTAssertNil(Fuzzy.rank(query: "cx", candidate: "abc"))               // order violated
    }
    func testPrefixBeatsScattered() {
        let a = Fuzzy.rank(query: "op", candidate: "Open PLY")!      // a contiguous prefix
        let b = Fuzzy.rank(query: "op", candidate: "Toggle Panel")!  // scattered (o..p)
        XCTAssertLessThan(a, b)
    }
    func testEmptyQueryMatchesAll() { XCTAssertNotNil(Fuzzy.rank(query: "", candidate: "x")) }
}
