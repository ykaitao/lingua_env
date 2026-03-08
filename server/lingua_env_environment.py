# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lingua Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import json
import random
import uuid
from pathlib import Path
import urllib.request

from openenv.core.env_server import Environment

from lingua_env.models import LinguaAction, LinguaObservation, LinguaState

data_path = Path("/tmp/ambiguous_examples.jsonl")
DATA_SOURCE = "https://huggingface.co/datasets/ykaitao/lingua_env_data/resolve/main/ambiguous_examples.jsonl"
if not data_path.exists():
    print(f"Downloading dataset from {DATA_SOURCE}")
    urllib.request.urlretrieve(DATA_SOURCE, data_path)

PYMORPHY_FEATURE_MAP = {
    "nomn": ("Case", "Nom"),
    "gent": ("Case", "Gen"),
    "datv": ("Case", "Dat"),
    "accs": ("Case", "Acc"),
    "ablt": ("Case", "Ins"),
    "loct": ("Case", "Loc"),
    "loc2": ("Case", "Loc"),
    "sing": ("Number", "Sing"),
    "plur": ("Number", "Plur"),
    "masc": ("Gender", "Masc"),
    "femn": ("Gender", "Fem"),
    "neut": ("Gender", "Neut"),
    "anim": ("Animacy", "Anim"),
    "inan": ("Animacy", "Inan"),
    "past": ("Tense", "Past"),
    "pres": ("Tense", "Pres"),
    "futr": ("Tense", "Fut"),
    "1per": ("Person", "1"),
    "2per": ("Person", "2"),
    "3per": ("Person", "3"),
    "COMP": ("Degree", "Cmp"),
    "Supr": ("Degree", "Sup"),
}

POS_MAP = {
    "NOUN": ["NOUN"],
    "ADJ": ["ADJF", "ADJS"],
    "VERB": ["VERB", "INFN", "GRND"],
    "AUX": ["VERB", "INFN"],
    "ADV": ["ADVB", "COMP", "PRED"],
    "PRON": ["NPRO"],
    "DET": ["ADJF", "NPRO"],
    "NUM": ["NUMR"],
    "ADP": ["PREP"],
    "CCONJ": ["CONJ"],
    "SCONJ": ["CONJ"],
    "PART": ["PRCL"],
    "INTJ": ["INTJ"],
}


def pymorphy_tag_to_ud_feats(tag_str: str) -> dict[str, str]:
    feats = {}
    for tok in tag_str.replace(",", " ").split():
        if tok in PYMORPHY_FEATURE_MAP:
            k, v = PYMORPHY_FEATURE_MAP[tok]
            feats[k] = v
    return feats


def parse_pymorphy_pos(tag_str: str) -> str:
    return tag_str.split()[0].split(",")[0]


def score_candidate(candidate: dict, gold: dict) -> dict:
    cand_lemma = candidate["lemma"]
    cand_tag = candidate["tag"]

    lemma_match = cand_lemma == gold["lemma"]

    pymorphy_pos = parse_pymorphy_pos(cand_tag)
    allowed_pos = POS_MAP.get(gold["upos"], [gold["upos"]])
    pos_match = pymorphy_pos in allowed_pos

    cand_feats = pymorphy_tag_to_ud_feats(cand_tag)
    gold_feats = gold.get("feats", {})

    reward = 0
    if lemma_match:
        reward += 1
    if pos_match:
        reward += 1

    feature_matches = {}
    for feat_name, gold_value in gold_feats.items():
        cand_value = cand_feats.get(feat_name)
        match = cand_value == gold_value
        feature_matches[feat_name] = {
            "candidate": cand_value,
            "gold": gold_value,
            "match": match,
        }
        if match:
            reward += 1

    return {
        "reward": reward,
        "lemma_match": lemma_match,
        "pos_match": pos_match,
        "candidate_feats": cand_feats,
        "gold_feats": gold_feats,
        "feature_matches": feature_matches,
    }


class LinguaEnvironment(Environment):
    def __init__(self):
        super().__init__()

        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f]

        self.current = None
        self._state = LinguaState(gold={})

    def reset(self) -> LinguaObservation:
        self.current = random.choice(self.examples)
        self._state = LinguaState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            gold=self.current["gold"],
        )

        return LinguaObservation(
            sentence=list(self.current["sentence"]),
            target_index=int(self.current["target_index"]),
            word=str(self.current["word"]),
            candidate_parses=list(self.current["candidate_parses"]),
        )

    def step(self, action: LinguaAction) -> LinguaObservation:
        if self.current is None:
            raise RuntimeError("Environment must be reset before step().")

        choice = int(action.choice)
        if choice < 0 or choice >= len(self.current["candidate_parses"]):
            raise ValueError(f"Invalid choice index: {choice}")

        self._state.step_count += 1

        chosen = self.current["candidate_parses"][choice]
        details = score_candidate(chosen, self.current["gold"])

        self._state.gold = self.current["gold"]

        obs = LinguaObservation(
            sentence=list(self.current["sentence"]),
            target_index=int(self.current["target_index"]),
            word=str(self.current["word"]),
            candidate_parses=list(self.current["candidate_parses"]),
        )
        obs.reward = float(details["reward"])
        obs.done = True
        obs.metadata = {
            "chosen": chosen,
            "gold": self.current["gold"],
            "reward_breakdown": details,
        }
        return obs

    @property
    def state(self) -> LinguaState:
        return self._state
