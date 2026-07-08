from agent import BudgetContextManager, WordTokenCounter, RankedResult, HeuristicCompactor


def test_budget_text_within_limit():
    mgr = BudgetContextManager(budgets={"memory": 100})
    result = mgr.budget_text("memory", "hello world")
    assert result == "hello world"
    assert mgr.report()["memory"] > 0


def test_budget_text_truncates():
    mgr = BudgetContextManager(budgets={"memory": 5})
    result = mgr.budget_text("memory", "hello world this is a very long string")
    assert "truncated" in result


def test_prepare_sources_empty():
    mgr = BudgetContextManager()
    result = mgr.prepare_sources([])
    assert result == []


def test_prepare_sources_with_data():
    mgr = BudgetContextManager()
    items = [RankedResult("t1", {"summary": "data"}, 1, "test")]
    result = mgr.prepare_sources(items)
    assert len(result) == 1


def test_report_keys():
    mgr = BudgetContextManager()
    mgr.budget_text("memory", "test mem")
    report = mgr.report()
    assert "memory" in report
    assert "tools" in report
    assert "total" in report


def test_reset_clears_usage():
    mgr = BudgetContextManager()
    mgr.budget_text("memory", "some text")
    assert mgr.report()["memory"] > 0
    mgr.reset()
    assert mgr.report()["memory"] == 0