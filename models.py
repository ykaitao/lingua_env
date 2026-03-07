# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Lingua Env Environment.

The lingua_env environment is a simple test environment that echoes back messages.
"""

from typing import Any

from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class LinguaAction(Action):
    choice: int


class LinguaObservation(Observation):
    sentence: list[str]
    target_index: int
    word: str
    candidate_parses: list[dict[str, Any]]


class LinguaState(State):
    gold: dict[str, Any] = Field(default_factory=dict)

