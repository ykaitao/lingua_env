from models import LinguaAction
from openenv.core.env_client import EnvClient

class LinguaClient(EnvClient[LinguaAction, dict, dict]):
    def _step_payload(self, action: LinguaAction) -> dict:
        return {"choice": action.choice}

    def _parse_result(self, payload: dict):
        return payload

    def _parse_state(self, payload: dict):
        return payload


with LinguaClient(base_url="http://127.0.0.1:8000") as client:
    reset_result = client.reset()
    print("RESET:")
    print(reset_result)

    step_result = client.step(LinguaAction(choice=0))
    print("STEP:")
    print(step_result)

    state_result = client.state()
    print("STATE:")
    print(state_result)