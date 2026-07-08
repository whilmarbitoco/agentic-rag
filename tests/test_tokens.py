from agent import WordTokenCounter


def test_word_token_counter_empty():
    c = WordTokenCounter()
    assert c.count("") == 0


def test_word_token_counter_short():
    c = WordTokenCounter()
    assert c.count("hello") == 1


def test_word_token_counter_long():
    c = WordTokenCounter()
    text = "x" * 200
    assert c.count(text) == 50


def test_word_token_counter_exact():
    c = WordTokenCounter()
    assert c.count("four") == 1
