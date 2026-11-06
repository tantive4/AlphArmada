"""Persistent headless worker with explicit natural actions and branchable game handles.

Check status.fullGameReady before relying on complete source-content coverage.
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable, Iterable


class HeadlessError(RuntimeError):
    pass


class InvalidSetupError(HeadlessError):
    """Discard this setup attempt; it has no winner or training reward."""

    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.retryable = bool(result.get("retryable"))
        super().__init__(result.get("message", "Invalid setup."))


class HeadlessClient:
    def __init__(self, assembly: str | Path | None = None, data: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        assembly = Path(assembly) if assembly is not None else root / "src" / "Swarmada.Headless.Cli" / "bin" / "Release" / "net9.0" / "Swarmada.Headless.Cli.dll"
        if not assembly.is_file():
            raise FileNotFoundError(f"Build the separate CLI project first: {assembly}")
        data = Path(data) if data is not None else root / "data"
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._failure: str | None = None
        self._stopped = False
        self._stderr_text = ""
        self._process = subprocess.Popen(
            ["dotnet", str(assembly.resolve()), "serve", "--data", str(data.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            self.status = self.request("status")
            if self.status.get("protocolVersion") != 1:
                raise HeadlessError("Unsupported headless protocol version.")
        except BaseException:
            self.close()
            raise

    def request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            if self._closed:
                raise HeadlessError("This headless client is closed.")
            if self._failure is not None:
                raise HeadlessError(self._failure)
            process = self._process
            if process.poll() is not None:
                raise self._transport_error("Headless worker exited")
            self._sequence += 1
            request = {**payload, "id": self._sequence, "op": operation}
            assert process.stdin is not None and process.stdout is not None
            encoded = json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n"
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
                line = process.stdout.readline()
            except (OSError, ValueError) as error:
                raise self._transport_error(f"Headless worker pipe failed: {error}") from error
            if not line:
                raise self._transport_error("Headless worker closed its output")
            response = json.loads(line)
            if response.get("id") != self._sequence:
                raise HeadlessError("Headless response ID does not match the request.")
            if not response.get("ok"):
                error = response.get("error", {})
                if error.get("invalidSetup") is not None:
                    raise InvalidSetupError(error["invalidSetup"])
                raise HeadlessError(f"{error.get('type', 'Error')}: {error.get('message', '')}")
            return response["result"]

    def resolve_attacks(self, requests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.request("attack_batch", requests=list(requests))

    def create_game(self, match: dict[str, Any], seed: int = 0) -> HeadlessGame:
        result = self.request("create_game", match=match, seed=seed)
        return HeadlessGame(self, result["gameId"], result["topology"], result["view"])

    def create_game_with_setup_retry(self, match: dict[str, Any] | Callable[[int], dict[str, Any]],
                                     complete_setup: Callable[[HeadlessGame, int], None],
                                     seeds: Iterable[int], maximum_attempts: int = 64) -> HeadlessGame:
        """Retry rejected setups with caller-supplied seeds and placement policy.

        The callback must finish setup or raise; rule errors are never retried.
        Rejected worker handles are closed before the next fresh match is created.
        """
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        last_error = None
        for attempt, seed in enumerate(islice(seeds, maximum_attempts), 1):
            game = self.create_game(match(seed) if callable(match) else match, seed)
            try:
                complete_setup(game, seed)
                if game.view.get("invalidSetup") is not None:
                    raise InvalidSetupError(game.view["invalidSetup"])
                if not game.view.get("setupComplete"):
                    raise HeadlessError("The setup policy returned before setup was complete.")
                game.setup_attempts = attempt
                return game
            except InvalidSetupError as error:
                game.close()
                if not error.retryable:
                    raise
                last_error = error
            except BaseException:
                game.close()
                raise
        if last_error is not None:
            raise last_error
        raise ValueError("The seed stream did not supply a setup attempt.")

    def starter_fleet(self, faction: str) -> dict[str, Any]:
        return self.request("starter_fleet", faction=faction)

    def validate_fleet(self, fleet: dict[str, Any], ruleset: str = "official-standard-400") -> dict[str, Any]:
        return self.request("validate_fleet", fleet=fleet, ruleset=ruleset)

    def import_fleet(self, exported_text: str) -> dict[str, Any]:
        return self.request("import_fleet", text=exported_text)

    def default_match(self, player_one_fleet: dict[str, Any], player_two_fleet: dict[str, Any],
                      objective_id: str = "targeting-beacons", first_player: int = 1) -> dict[str, Any]:
        return self.request("default_match", playerOneFleet=player_one_fleet, playerTwoFleet=player_two_fleet,
                            objectiveId=objective_id, firstPlayerId=first_player)

    def prepare_match(self, match: dict[str, Any], seed: int = 0) -> dict[str, Any]:
        """Prepare initial state; this does not execute gameplay or start self-play."""
        return self.request("prepare_match", match=match, seed=seed)

    def _transport_error(self, message: str) -> HeadlessError:
        self._failure = message
        detail = self._stop_worker(capture_stderr=True)
        self._failure = f"{message}: {detail.strip()}" if detail.strip() else message
        return HeadlessError(self._failure)

    @staticmethod
    def _close_pipe(stream: Any) -> None:
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except (OSError, ValueError):
                # Windows can report an invalid pipe during a buffered close after
                # the worker exits. Other pipes and the process must still be reaped.
                pass

    def _stop_worker(self, capture_stderr: bool = False) -> str:
        if self._stopped:
            return self._stderr_text
        self._stopped = True
        process = self._process
        try:
            self._close_pipe(process.stdin)
            exited = False
            try:
                process.wait(timeout=5)
                exited = True
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass  # The process may have exited between wait and kill.
                try:
                    process.wait(timeout=5)
                    exited = True
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except OSError:
                pass
            # Reading a live worker's stderr to EOF could block indefinitely.
            if capture_stderr and exited and process.stderr is not None:
                try:
                    self._stderr_text = process.stderr.read()
                except (OSError, ValueError):
                    pass
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                self._close_pipe(stream)
        return self._stderr_text

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._stop_worker()

    def __enter__(self) -> HeadlessClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class HeadlessGame:
    """A worker-owned match. Close unused search branches to release their memory.

    Positions use millimetres. Returned action templates may accept a continuous
    position/yaw or course, as declared by the enclosing option.
    """

    def __init__(self, client: HeadlessClient, game_id: int, topology: dict[str, Any], view: dict[str, Any]):
        self.client = client
        self.game_id = game_id
        self.topology = topology
        self.view = view
        self.setup_attempts = 1
        self._closed = False
        self.discrete_view: dict[str, Any] | None = None

    @property
    def invalid_setup(self) -> dict[str, Any] | None:
        return self.view.get("invalidSetup") or (self.discrete_view or {}).get("invalidSetup")

    def _request(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise HeadlessError("This game handle is closed.")
        return self.client.request(operation, gameId=self.game_id, **payload)

    def decision(self) -> dict[str, Any]:
        self.view = self._request("game_decision")
        return self.view["decision"]

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        self.discrete_view = None
        try:
            self.view = self._request("game_apply", action=action)
        except InvalidSetupError:
            try:
                self.decision()
            except HeadlessError:
                pass  # Preserve the typed rejection if its view refresh loses the worker.
            raise
        return self.view["decision"]

    def discrete_decision(self, candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Mask caller-supplied {id, action} entries without choosing an action.

        IDs and input order are preserved, including illegal entries. Spatial/course
        actions must be fully specified. Empty masks do not invent a fallback.
        """
        result = self._request("game_discrete_decision", candidates=list(candidates))
        self.view = result["view"]
        self.discrete_view = result["discreteDecision"]
        if self.discrete_view.get("invalidSetup") is not None:
            raise InvalidSetupError(self.discrete_view["invalidSetup"])
        return self.discrete_view

    def apply_discrete(self, action_id: int, decision_token: int | None = None) -> dict[str, Any]:
        """Apply exactly the chosen ID from the latest mask, with stale-set checks."""
        if decision_token is None:
            if self.discrete_view is None:
                raise HeadlessError("Request a discrete candidate mask before selecting an action ID.")
            decision_token = self.discrete_view["decisionToken"]
        self.view = self._request("game_apply_discrete", actionId=action_id, decisionToken=decision_token)
        self.discrete_view = None
        return self.view["decision"]

    def sample_chance(self) -> dict[str, Any]:
        self.discrete_view = None
        result = self._request("game_chance")
        self.view = result["view"]
        return {"action": result["action"], "decision": self.view["decision"]}

    def snapshot(self) -> dict[str, Any]:
        return self._request("game_state")

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.discrete_view = None
        self.view = self._request("game_restore", snapshot=snapshot)
        return self.view["decision"]

    def clone(self) -> HeadlessGame:
        result = self._request("game_clone")
        return HeadlessGame(self.client, result["gameId"], self.topology, result["view"])

    def benchmark(self, path: Iterable[dict[str, Any]] = (), iterations: int = 10000) -> dict[str, Any]:
        """Measure native branch work without changing this handle; excludes IPC and inference."""
        return self._request("game_benchmark", path=list(path), iterations=iterations)

    def request_time_limit(self) -> dict[str, Any]:
        """Finish the current round, then score the default tournament game."""
        self.discrete_view = None
        self.view = self._request("game_time_limit")
        return self.view["decision"]

    def concede(self, player: int) -> dict[str, Any]:
        self.discrete_view = None
        self.view = self._request("game_concede", player=player)
        return self.view["decision"]

    def close(self) -> None:
        if not self._closed:
            try:
                if not self.client._closed and self.client._failure is None:
                    self._request("game_close")
            except HeadlessError:
                if self.client._failure is None and not self.client._closed:
                    raise
            finally:
                self._closed = True

    def __enter__(self) -> HeadlessGame:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
