import pytest

from lab.model_limits import Decoder, layout_inventory


def model():
    return Decoder(linear_parameters=360_710_144, layers=18,
                   query_width=2048, kv_bytes_per_token=18_432)


def test_single_causal_chain_cannot_use_four_replicas():
    decoder = model()
    one = decoder.schedule([(157, 3252)], 1, 1, gpus=1)
    four = decoder.schedule([(157, 3252)], 1, 1, gpus=4)
    assert one["seconds"] == four["seconds"]
    assert max(d["passes"] for d in four["devices"]) == 3251


def test_independent_sequences_run_concurrently():
    decoder = model()
    single = decoder.schedule([(157, 100)], 1, 1, gpus=1, slots=1)
    four = decoder.schedule([(157, 100)] * 4, 1, 1, gpus=4, slots=1)
    serial = decoder.schedule([(157, 100)] * 4, 1, 1, gpus=1, slots=1)
    assert four["seconds"] == single["seconds"]
    assert serial["seconds"] == pytest.approx(4 * single["seconds"])


def test_prefill_emits_first_token_and_empty_decode_is_free():
    decoder = model()
    assert decoder.schedule([(157, 1)], 1, 1)["seconds"] == 0
    assert decoder.schedule([], 1, 1)["seconds"] == 0
    assert decoder.schedule([(157, 2)], 1, 1)["devices"][0]["passes"] == 1


def test_speculation_shares_weights_but_pays_for_all_verification_positions():
    decoder = model()
    greedy = decoder.schedule([(157, 101)], 1, 1, gpus=1)
    perfect = decoder.schedule([(157, 101)], 5, 5, gpus=1)
    partial = decoder.schedule([(157, 101)], 2, 5, gpus=1)
    assert perfect["devices"][0]["passes"] == 20
    assert partial["devices"][0]["passes"] == 50
    assert perfect["flops"] == greedy["flops"]
    assert partial["flops"] > greedy["flops"]
    assert perfect["seconds"] < partial["seconds"] < greedy["seconds"]
    # Five positions add KV writes: the speedup is slightly less than five.
    assert perfect["seconds"] > greedy["seconds"] / 5


def test_batch_reuses_matrix_weights_and_accounts_for_each_kv_cache():
    decoder = model()
    _, _, single_bytes = decoder.step([(100, 10)], 1)
    _, _, batch_bytes = decoder.step([(100, 10)] * 8, 1)
    assert batch_bytes == 2 * decoder.linear_parameters + 8 * 101 * decoder.kv_bytes_per_token
    assert batch_bytes < 8 * single_bytes


def test_layout_grouped_convolution_counts_filter_input_channels_once():
    def output(identifier, shape):
        return {"%": identifier, "TT": {"D": [{"#": "0.t_f32"}, shape]}}

    ops = [
        {"#": "input", "O": [output(1, [-1, 4, 5, 5])]},
        {"#": "p", "O": output(2, [4, 1, 3, 3])},
        {"#": "1.depthwise_conv2d", "I": [{"%": 1}, {"%": 2}],
         "O": [output(3, [-1, 4, 3, 3])], "A": []},
    ]
    graph = {"program": {"regions": [{"blocks": [{"ops": ops}]}]}}
    result = layout_inventory(graph)
    assert result["flops_per_page"] == 2 * 4 * 3 * 3 * 1 * 3 * 3
    ops[-1]["O"][0]["TT"]["D"][1][-1] = -1
    with pytest.raises(ValueError, match="Unresolved"):
        layout_inventory(graph)
