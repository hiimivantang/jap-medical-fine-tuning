from jmedft.reporting.costs import openai_cost_per_1k_requests


def test_openai_cost_per_1k_requests_hand_computed():
    # gpt-4.1 list: $2.00 / 1M input, $8.00 / 1M output. 500 in + 40 out per request:
    # per request = 500*2e-6 + 40*8e-6 = 0.001 + 0.00032 = 0.00132 → per 1k = 1.32
    prices = {"input_per_mtok": 2.00, "output_per_mtok": 8.00}
    assert abs(openai_cost_per_1k_requests(prices, 500, 40) - 1.32) < 1e-9
