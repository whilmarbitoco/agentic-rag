from agent import HeuristicCompactor, RankedResult, WordTokenCounter


def _rr(text, score=1):
    return RankedResult("t1", {"body": text}, score, "test")


def test_compactor_empty():
    c = HeuristicCompactor()
    result = c.compact([], 1000, WordTokenCounter())
    assert result == []


def test_compactor_keeps_all_under_budget():
    c = HeuristicCompactor()
    items = [_rr("short", 2), _rr("tiny", 1)]
    result = c.compact(items, 5000, WordTokenCounter())
    assert len(result) == 2


def test_compactor_truncates_overflow():
    c = HeuristicCompactor()
    items = [_rr("x" * 4000, 2)]
    result = c.compact(items, 50, WordTokenCounter())
    assert len(result) == 1
    assert "truncated" in result[0].data.get("summary", "") or "truncated" in str(result[0].data)


def test_compactor_drops_overflow_items():
    c = HeuristicCompactor()
    items = [_rr("x" * 200, 2), _rr("y" * 200, 1)]
    result = c.compact(items, 10, WordTokenCounter())
    assert len(result) <= 1


def test_compactor_prefers_highest_score():
    c = HeuristicCompactor()
    high = _rr("high value data", 3)
    low = _rr("low value data", 1)
    result = c.compact([low, high], 500, WordTokenCounter())
    assert len(result) >= 2
    assert result[0].score >= result[-1].score
