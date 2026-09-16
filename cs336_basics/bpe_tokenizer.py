from dataclasses import dataclass
from typing import Iterator
import regex as re
import time

def split_special_tokens(text: str, special_tokens: list[str]) -> Iterator[str]:
    if not special_tokens:
        yield text
    else:
        regex = re.escape("|".join(special_tokens))
        yield from re.split(regex, text)

def pretokenize(text: str) -> Iterator[str]:
    REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for pretoken in re.finditer(REGEX, text):
        yield pretoken.group(0)

def merge_tokens(tokens: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    """Find the merge with the lowest index and replace all those pairs with the merge"""
    while len(tokens) >= 2:
        min_pair = min(zip(tokens, tokens[1:]), key=lambda k: merges.get(k, float("inf")))
        if min_pair not in merges:
            break
    
        new_tokens = []
        while len(tokens) >= 2:
            pair = tuple(tokens[:2])
            if pair == min_pair:
                tokens = [merges[pair]] + tokens[2:]
            else:
                new_tokens.append(tokens.pop(0))
        tokens = new_tokens + tokens

    return tokens


@dataclass
class PairData:
    pair_counts: dict[tuple[int, int], int]
    pretoken_counts: dict[tuple[int, ...], int]
    pair_to_pretokens: dict[tuple[int, int], set[tuple[int, ...]]]

    @staticmethod
    def create(text: str, special_tokens: list[str], merges: dict[tuple[int, int], int]) -> "PairData":
        pair_counts: dict[tuple[int, int], int] = {}
        pretoken_counts: dict[tuple[int, ...], int] = {}
        pair_to_pretokens: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for part in split_special_tokens(text, special_tokens):
            for pretoken in pretokenize(part):
                pretoken_ids = list(pretoken.encode("utf-8"))
                pretoken_ids = tuple(merge_tokens(tokens=pretoken_ids, merges=merges))
                for pair in zip(pretoken_ids, pretoken_ids[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    pair_to_pretokens[pair] = pair_to_pretokens.get(pair, set())| {pretoken_ids}
                pretoken_counts[pretoken_ids] = pretoken_counts.get(pretoken_ids, 0) + 1
        return PairData(pair_counts=pair_counts, pretoken_counts=pretoken_counts, pair_to_pretokens=pair_to_pretokens)

    def update_merged_pair(self, pair: tuple[int, int], merges: dict[tuple[int, int], int]) -> int:
        """Updates the data when we merge a pair.
        Given axyb and merge xy=z, find and decrement ax and yb, increment az and zb
        """
        vocab_idx = merges[pair]
        for pretoken_ids in self.pair_to_pretokens[pair]:
            if pretoken_ids not in self.pretoken_counts:
                continue

            new_pretoken_ids = tuple(merge_tokens(tokens=list(pretoken_ids), merges=merges))
            pretoken_count = self.pretoken_counts.pop(pretoken_ids)
            self.pretoken_counts[new_pretoken_ids] = pretoken_count
            for idx, curr_pair in enumerate(zip(pretoken_ids, pretoken_ids[1:])):
                if curr_pair == pair:
                    prev_pair = pretoken_ids[idx-1: idx+1] if idx != 0 else []
                    next_pair = pretoken_ids[idx+1: idx+3]
                    if len(prev_pair) == 2:
                        self.pair_counts[prev_pair] -= pretoken_count
                        new_prev_pair = (prev_pair[0], vocab_idx)
                        self.pair_counts[new_prev_pair] = self.pair_counts.get(new_prev_pair, 0) + pretoken_count
                        self.pair_to_pretokens[new_prev_pair] = self.pair_to_pretokens.get(new_prev_pair, set()) | {new_pretoken_ids}
                    if len(next_pair) == 2:
                        self.pair_counts[next_pair] -= pretoken_count
                        new_next_pair = (vocab_idx, next_pair[1])
                        self.pair_counts[new_next_pair] = self.pair_counts.get(new_next_pair, 0) + pretoken_count
                        self.pair_to_pretokens[new_next_pair] = self.pair_to_pretokens.get(new_next_pair, set()) | {new_pretoken_ids}
        return self.pair_counts.pop(pair)


class BPETokenizer:
    NUM_BYTE_VALUES: int = 256

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}

    def train(self, text: str, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab = {idx: bytes([idx]) for idx in range(self.NUM_BYTE_VALUES)}

        for vocab_idx in range(len(self.vocab), vocab_size - len(special_tokens)):
            pair_data = PairData.create(text=text, special_tokens=special_tokens, merges=self.merges)
            pair_counts = pair_data.pair_counts
            max_pair = max(pair_counts.keys(), key=lambda k: (pair_counts[k], k))
            self.vocab[vocab_idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.merges[max_pair] = vocab_idx

        # NOTE: special tokens usually added at end
        for special_token in special_tokens:
            self.special_tokens[special_token] = len(self.vocab)
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")


    def train_from_file(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:
        with open(input_path, "r") as f:
            text = f.read()
        self.train(text=text, vocab_size=vocab_size, special_tokens=special_tokens)

    def get_byte_merges(self) -> dict[tuple[bytes, bytes], int]:
        return {(self.vocab[k[0]], self.vocab[k[1]]): v for k, v in self.merges.items()}

    def get_vocab(self) -> dict[int, bytes]:
        return self.vocab


class FastBPETokenizer(BPETokenizer):
    def train(self, text: str, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab = {idx: bytes([idx]) for idx in range(self.NUM_BYTE_VALUES)}

        pair_data = PairData.create(text=text, special_tokens=special_tokens, merges=self.merges)
        for vocab_idx in range(len(self.vocab), vocab_size - len(special_tokens)):
            pair_counts = pair_data.pair_counts
            max_pair = max(pair_counts.keys(), key=lambda k: (pair_counts[k], k))
            self.vocab[vocab_idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.merges[max_pair] = vocab_idx
            pair_data.update_merged_pair(pair=max_pair, merges=self.merges)

        # NOTE: special tokens usually added at end
        for special_token in special_tokens:
            self.special_tokens[special_token] = len(self.vocab)
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")

                

if __name__ == "__main__":
    for tokenizer in [BPETokenizer(), FastBPETokenizer()]:
        text = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest" * 100
        a = time.time()
        tokenizer.train(text, 264, [" ", "\n"])
        print(time.time() - a, tokenizer.get_byte_merges())
