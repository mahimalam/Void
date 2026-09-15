"""Configuration loader.

Parses config.yaml into typed dataclasses so the rest of the codebase
gets autocomplete and KeyError-free access.

Secrets (API keys, tokens) are NOT stored in config.yaml. They live in
``data/.env`` (gitignored) as ``KEY=value`` pairs. The loader resolves
``${ENV_VAR_NAME}`` placeholders in config.yaml against that file and
the process environment. See ``data/.env.example`` for the schema.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.errors import ConfigError

_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _log():
    """Late-bound logger to avoid the core.config <-> core.logging_setup
    circular import that bites if we grab the logger at module load time.
    """
    from core.logging_setup import get_logger
    return get_logger("config")




def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser. Supports ``KEY=VALUE`` and ``export KEY=VALUE``.

    - Strips optional surrounding single or double quotes.
    - Skips blank lines and lines starting with ``#``.
    - Ignores malformed lines with a warning (does not raise).

    No new dependency: ``python-dotenv`` is intentionally avoided to keep
    the install footprint small.
    """
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            _log().warning("env_file_skip_malformed", line=line[:60])
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            _log().warning("env_file_skip_empty_key", line=line[:60])
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def _load_secrets(project_root: Path) -> dict[str, str]:
    """Merge secrets from ``data/.env`` (preferred) and process environment.

    ``data/.env`` wins on conflict for explicit lookups; the process
    environment is the fallback. Either source alone is enough — callers
    can leave the file out and export variables in the shell.
    """
    env_file = project_root / "data" / ".env"
    file_vars = _parse_env_file(env_file)
    merged: dict[str, str] = dict(os.environ)
    for k, v in file_vars.items():
        merged.setdefault(k, v)
    return merged


def _resolve_secret(raw: Any, secrets: dict[str, str], *, field_name: str) -> Any:
    """Resolve a config value that may be a ``${ENV_VAR}`` placeholder.

    Behaviour:
    - ``None`` / empty list / empty string → returned unchanged so
      ``enabled: false`` style configs still work.
    - ``"${FOO}"`` → looked up in ``secrets``. Missing raises
      ``ConfigError`` with an actionable message.
    - Anything else (a literal string or a list of strings) → returned
      unchanged, with a one-time warning that hardcoding secrets in
      config.yaml is deprecated. The warning is emitted at most once
      per field to avoid log spam.
    """
    if raw is None:
        return raw
    if isinstance(raw, str):
        m = _PLACEHOLDER_RE.match(raw)
        if m:
            var = m.group(1)
            if var not in secrets or not secrets[var]:
                raise ConfigError(
                    f"Secret placeholder ${{{var}}} referenced by {field_name} is "
                    f"not set. Add {var}=... to data/.env or export it in the shell.",
                    spoken="A required secret is missing from data/.env.",
                )
            return secrets[var]
        if raw.strip():
            _log().warning(
                "secret_in_config_yaml_deprecated",
                field=field_name,
                hint="Move the value to data/.env and use ${ENV_VAR_NAME}.",
            )
        return raw
    if isinstance(raw, list):
        resolved: list[str] = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                m = _PLACEHOLDER_RE.match(item)
                if m:
                    var = m.group(1)
                    if var not in secrets or not secrets[var]:
                        raise ConfigError(
                            f"Secret placeholder ${{{var}}} referenced by {field_name}[{i}] "
                            f"is not set. Add {var}=... to data/.env or export it in the shell.",
                            spoken="A required secret is missing from data/.env.",
                        )
                    resolved.append(secrets[var])
                else:
                    if item.strip():
                        _log().warning(
                            "secret_in_config_yaml_deprecated",
                            field=f"{field_name}[{i}]",
                            hint="Move the value to data/.env and use ${ENV_VAR_NAME}.",
                        )
                    resolved.append(item)
            else:
                resolved.append(item)
        return resolved
    return raw


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    renderer: str = "pretty"
    log_file: str = "data/jarvis.log"
    log_to_stdout: bool = True


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_in: int = 16000
    sample_rate_out: int = 22050
    input_device: str | int | None = None
    output_device: str | int | None = None
    frame_ms: int = 80
    channels: int = 1
    gain: float = 1.0

    @property
    def frame_samples_in(self) -> int:
        return int(self.sample_rate_in * self.frame_ms / 1000)


@dataclass(frozen=True)
class WakeWordConfig:
    model_path: str = "data/wakeword/jarvis.onnx"
    fallback_builtin: str = "hey_jarvis"
    threshold: float = 0.5
    cooldown_ms: int = 1500
    keyboard_trigger: bool = False


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    speech_pad_ms: int = 250
    min_silence_ms: int = 1100
    min_speech_ms: int = 240
    barge_in_min_ms: int = 300
    max_utterance_ms: int = 30000
    initial_silence_timeout_s: float = 5.5


@dataclass(frozen=True)
class STTConfig:
    model: str = "turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    # 2026-06-20: STT multilingual upgrade for the user's
    # Bengali accent. ``primary_language`` is what we tell
    # faster-whisper the audio is in (default "en" — the
    # user's commands are English, just heavily accented).
    # If Whisper's auto-detected language is non-English
    # with high confidence, we retry with
    # ``task="translate"`` which forces an English output
    # no matter what the source language was — the user
    # can speak Bengali, Hindi, or code-mixed Banglish and
    # JARVIS will still get English text to feed to the LLM.
    primary_language: str = "en"
    multilingual_fallback: bool = True
    # Average per-segment logprob below this triggers the
    # multilingual fallback (default -0.6 ≈ 55% confidence
    # per token, a reasonable "I'm not sure" threshold).
    min_confidence: float = -0.6
    # Per-token no-speech probability above this means the
    # model thinks a chunk was silence; we use it to reject
    # hallucinated outputs (Whisper sometimes invents text on
    # pure silence when beam search is greedy).
    no_speech_threshold: float = 0.6
    # Whisper task: "transcribe" keeps the source language,
    # "translate" forces English output. The fallback path
    # uses "translate" to recover from Bengali-accented
    # English that gets misclassified as Bengali script.
    task: str = "transcribe"
    beam_size: int = 1
    vocabulary_file: str = "data/vocabulary.txt"
    # Whisper initial_prompt tweaks. The prompt biases the
    # model's first-token predictions toward the listed
    # context, dramatically improving recognition of names
    # and uncommon terms. We compose:
    #   1. A static context line (locale + persona).
    #   2. The user's vocabulary.txt (one term per line).
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    static_prompt: str = (
        "J.A.R.V.I.S. voice assistant. The user is from Bangladesh "
        "and speaks English with a Bengali accent. Common terms: "
        "FIFA, cricket, IPL, match schedule, Jarvis, Mahim, Dhaka, "
        "Bangladesh."
    )


@dataclass(frozen=True)
class BrainModel:
    name: str
    estimated_vram_gb: float


@dataclass(frozen=True)
class GeminiConfig:
    enabled: bool = False
    api_key: str | None = None
    # Multi-key pool: GEMINI_API_KEY_1, _2, _3 ... in data/.env
    # Each Google account = 250 req/day free. Pool rotates proactively.
    api_keys: list[str] = field(default_factory=list)
    model_flash: str = "stepfun/step-3.7-flash:free"
    model_pro: str = "nvidia/nemotron-3-super-120b-a12b:free"
    model_grok: str = "xai.grok-4.3"
    model_embed: str = "cohere.embed-v4.0"
    project: str | None = None
    location: str = "us-central1"
    service_account_json: str | None = None
    timeout_seconds: int = 10
    use_pro: bool = True
    base_url: str = "https://api.kilo.ai/api/gateway/v1"


@dataclass(frozen=True)
class RoutingMode:
    """Constants for the brain-router dispatch mode.

    - ``local_first``: Run complexity classifier on the local Qwen brain.
      SIMPLE → Qwen generates locally. COMPLEX → Gemini in the cloud.
      On cloud failure, fall back to local Qwen. On local OOM, fall back
      to SmolLM. This is the project default and restores the
      "local-first, cloud fallback for hard queries" promise.
    - ``cloud_first``: Send every request to Gemini when it is configured
      and available. Local Qwen is the fallback. Original behaviour before
      H1 was fixed.
    - ``local_only``: Never touch the cloud. Always run the local
      classifier and use the local Qwen. On OOM, fall back to SmolLM.
    - ``cloud_only``: Always Gemini. No local fallback. Useful for users
      who explicitly want the cloud behaviour and accept the API cost.
    """
    LOCAL_FIRST = "local_first"
    CLOUD_FIRST = "cloud_first"
    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"


_ROUTING_MODES = frozenset({
    RoutingMode.LOCAL_FIRST,
    RoutingMode.CLOUD_FIRST,
    RoutingMode.LOCAL_ONLY,
    RoutingMode.CLOUD_ONLY,
})


@dataclass(frozen=True)
class BrainsConfig:
    routing_mode: str = RoutingMode.LOCAL_FIRST
    cloud_fallback_enabled: bool = True
    brain1_primary: BrainModel = field(default_factory=lambda: BrainModel(name="qwen3:4b-instruct", estimated_vram_gb=2.5))
    brain1_fallback: BrainModel = field(default_factory=lambda: BrainModel(name="qwen2.5:3b-instruct-q4_K_M", estimated_vram_gb=1.9))
    brain2_specialist: GeminiConfig = field(default_factory=GeminiConfig)
    brain3_failsafe: BrainModel = field(default_factory=lambda: BrainModel(name="smollm3:135m-instruct-q4_K_M", estimated_vram_gb=0.5))


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 60
    keep_alive: str = "30m"
    num_ctx: int = 4096
    temperature: float = 0.7
    startup_probe_tokens: int = 4
    # H34: which Gemini model to use for memory compaction and the
    # weekly observer. ``flash`` (default) is fast and cheap, but
    # ``pro`` produces noticeably better 3-sentence summaries and
    # workstyle.md analyses. Set to ``flash`` in config.yaml for
    # the cheapest path; flip to ``pro`` for higher-quality
    # extraction.
    compaction_brain: str = "flash"


@dataclass(frozen=True)
class TTSConfig:
    binary: str = "piper"
    voice_model: str = "data/piper/en_US-lessac-medium.onnx"
    voice_config: str = "data/piper/en_US-lessac-medium.onnx.json"
    worker_count: int = 2
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sample_rate: int = 22050
    # Deepgram Aura-2 TTS — primary cloud voice.
    # Resolved from data/.env via DEEPGRAM_API_KEY.
    # $200 free credit on signup, then $0.015/1000 chars.
    # Wake-phrase voice switching:
    #   "Jarvis"     → aura-2-andromeda-en (professional, authoritative)
    #   "Hey Jarvis" → aura-2-amalthea-en  (warm, expressive)
    deepgram_api_key: str | None = None


@dataclass(frozen=True)
class StreamingConfig:
    min_chunk_chars: int = 30
    max_chunk_chars: int = 240
    sentence_terminators: str = r"[.!?]+"


@dataclass(frozen=True)
class FollowUpConfig:
    enabled: bool = False
    timeout_s: float = 8.0
    headphones_only_warning: bool = True
    # 2026-06-20: continuous-loop follow-up. ``idle_timeout_s``
    # is how long the loop waits for the next utterance before
    # giving up and going back to wake-word IDLE.
    # ``session_timeout_s`` is the hard cap on total time
    # inside the loop. ``sleep_phrases`` is the list of
    # user-said phrases that end the loop early.
    idle_timeout_s: float = 30.0
    session_timeout_s: float = 600.0
    sleep_phrases: tuple[str, ...] = (
        "go to sleep",
        "stop listening",
        "thanks that's all",
        "thank you that's all",
        "that'll be all",
        "that will be all",
        "jarvis stop",
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    unrecoverable_error_apology: str = "I ran into a problem and had to reset. Try again."
    state_idle_timeout_ms: int = 0
    # Post-speak settle window (H31). The orchestrator waits this long
    # AFTER the audio output is fully drained, to let the speaker
    # hardware buffer and any room reverb decay before the wake-word
    # detector goes live again. The previous hardcoded 1000 ms was the
    # biggest source of per-turn latency. Default 300 ms is enough for
    # a wired speaker + quiet room; bump it for Bluetooth speakers or
    # noisy rooms. Set to 0 to disable the settle window entirely
    # (only safe if AEC is reliable).
    post_speak_settle_ms: int = 300
    # 2026-06-20: barge-in minimum sustained speech (ms). The
    # barge-in monitor runs VAD on the AEC-processed mic stream
    # during TTS playback; this is how many consecutive speech
    # frames (at 80 ms each) are required before interrupting.
    # 320 ms ≈ 4 frames — short enough that a single word like
    # "wait" or "stop" interrupts, long enough to ignore coughs,
    # laughs, and transient spikes. Wake-word detection is a
    # parallel backup signal (always-on).
    barge_in_min_speech_ms: int = 320
    # 2026-06-20: minimum STT confidence (avg per-segment
    # logprob) for follow-up loop transcripts. The follow-up
    # loop is the most barge-in-vulnerable state because
    # TTS bleed gets captured there too. Discard transcripts
    # below this threshold without bothering the LLM. The
    # default -1.0 is a reasonable "hallucination floor" —
    # normal commands typically score above -0.5.
    follow_up_min_confidence: float = -1.0
    # 2026-06-20: TTS duck volume during suspected user
    # speech. The moment VAD fires, the speaker is muted to
    # this fraction (default 0.1 = -20 dB) which kills the
    # TTS bleed that's the #1 cause of false barge-ins on
    # laptop built-in speakers. 0.0 = full mute (silent
    # while user talks), 1.0 = no ducking (defeats the
    # purpose). 0.1 leaves the user with audible context.
    barge_in_duck_volume: float = 0.1
    # 2026-06-20: push-to-talk enable flag. When true,
    # the orchestrator installs a Windows low-level keyboard
    # hook on Ctrl+Space. Pressing it during TTS playback
    # triggers an immediate barge-in (always works,
    # regardless of audio quality). Set to false to disable
    # the hook (e.g. if it conflicts with other software).
    push_to_talk_enabled: bool = True
    # First-run voiceprint enrollment timeout (H24). The previous
    # behavior blocked the voice loop indefinitely if the user did
    # not speak during the 10 s capture window. Default 30 s gives
    # a generous window while still letting the user say "skip it"
    # by being silent. On timeout the partial voiceprint (if any)
    # is discarded and the voice loop starts immediately. The user
    # can re-enroll later via the ``re_enroll_voiceprint`` tool.
    enrollment_timeout_s: float = 30.0
    # Observer background-task shutdown grace (H23). On shutdown the
    # orchestrator waits up to this long for the in-flight weekly
    # observer to write its workstyle.md before giving up. Default
    # 5 s; the observer typically completes in 1-3 s, so this is
    # purely a safety net. Set to 0 to skip the wait entirely
    # (next observer run will overwrite the partial file).
    observer_shutdown_grace_s: float = 5.0
    follow_up: FollowUpConfig = field(default_factory=FollowUpConfig)


@dataclass(frozen=True)
class PromptsConfig:
    voice_mode: str = "prompts/voice_mode.txt"
    tool_chaining: str = "prompts/tool_chaining.txt"
    complexity_classifier: str = "prompts/complexity_classifier.txt"
    difficulty_classifier: str = "prompts/difficulty_classifier.txt"


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    runtime_state: str = "data/runtime_state.json"
    audit_log_db: str = "data/jarvis.db"
    workstyle: str = "data/workstyle.md"


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool = False
    unlock_timeout_minutes: int = 15


@dataclass(frozen=True)
class AECConfig:
    enabled: bool = True
    filter_length: int = 32000  # ring buffer size in samples (2s at 16kHz)
    delay_ms: int = 25          # acoustic travel delay: speaker → mic
    mic_sample_rate: int = 16000
    tts_sample_rate: int = 22050


@dataclass(frozen=True)
class ConversationConfig:
    enabled: bool = True
    max_turns: int = 20  # keep last 20 turns (10 full exchanges)


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9090


@dataclass(frozen=True)
class SearchConfig:
    tavily_api_key: list[str] | str | None = None
    serper_api_key: list[str] | str | None = None


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = True
    max_calls_per_turn: int = 0  # 0 or <= 0 means unlimited tool calls
    execution_timeout_s: int = 30
    sandbox_dir: str = "data/sandbox"
    notes_dir: str = "data/notes"


@dataclass(frozen=True)
class Config:
    project_root: Path
    logging: LoggingConfig
    audio: AudioConfig
    wake_word: WakeWordConfig
    vad: VADConfig
    stt: STTConfig
    brains: BrainsConfig
    ollama: OllamaConfig
    tts: TTSConfig
    streaming: StreamingConfig
    orchestrator: OrchestratorConfig
    prompts: PromptsConfig
    paths: PathsConfig
    verification: VerificationConfig
    aec: AECConfig
    conversation: ConversationConfig
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def resolve(self, relative_or_abs: str | Path) -> Path:
        """Resolve a config-supplied path against the project root."""
        p = Path(relative_or_abs)
        return p if p.is_absolute() else (self.project_root / p)


def _brain_model(d: dict[str, Any]) -> BrainModel:
    return BrainModel(name=str(d["name"]), estimated_vram_gb=float(d["estimated_vram_gb"]))


def load_config(path: str | Path = "config.yaml", project_root: Path | None = None) -> Config:
    cfg_path = Path(path)
    if not cfg_path.is_absolute() and project_root is not None:
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}", spoken="Configuration file is missing.")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error: {e}", spoken="Configuration file is malformed.")

    root = project_root or cfg_path.parent.resolve()
    secrets = _load_secrets(root)

    try:
        # Resolve all secrets FIRST before building config objects.
        deepgram_api_key: str | None = secrets.get("DEEPGRAM_API_KEY") or None

        # Gemini pool keys — accepts any of these formats in data/.env:
        #   GEMINI_API_KEY_1, GEMINI_API_KEY_2  (with underscore)
        #   GEMINI_API_KEY1, GEMINI_API_KEY2    (without underscore)
        #   GEMINI_API_KEY                      (legacy single key)
        gemini_pool_keys: list[str] = []
        seen: set[str] = set()
        for i in range(1, 10):
            for fmt in (f"GEMINI_API_KEY_{i}", f"GEMINI_API_KEY{i}"):
                k = secrets.get(fmt, "").strip()
                if k and k not in seen:
                    gemini_pool_keys.append(k)
                    seen.add(k)
        # Legacy single key — add if not already in pool
        legacy = secrets.get("GEMINI_API_KEY", "").strip()
        if legacy and legacy not in seen:
            gemini_pool_keys.insert(0, legacy)
            seen.add(legacy)

        brains_raw = raw["brains"]
        gemini_raw = dict(brains_raw.get("brain2_specialist", {}))
        # Remove api_key placeholder from yaml — keys come from .env pool
        gemini_raw.pop("api_key", None)
        # Inject the primary key (first pool key) for backward compat
        if gemini_pool_keys:
            gemini_raw["api_key"] = gemini_pool_keys[0]
        gemini_raw["api_keys"] = gemini_pool_keys

        search_raw = dict(raw.get("search", {}))
        for key in ("tavily_api_key", "serper_api_key"):
            if key in search_raw:
                search_raw[key] = _resolve_secret(
                    search_raw[key], secrets, field_name=f"search.{key}",
                )
        routing_mode = brains_raw.get("routing_mode", RoutingMode.LOCAL_FIRST)
        if routing_mode not in _ROUTING_MODES:
            raise ConfigError(
                f"brains.routing_mode must be one of {sorted(_ROUTING_MODES)}, got {routing_mode!r}",
                spoken="Routing mode in config is invalid.",
            )
        cloud_fallback_enabled = bool(brains_raw.get("cloud_fallback_enabled", True))

        cfg = Config(
            project_root=root,
            logging=LoggingConfig(**raw.get("logging", {})),
            audio=AudioConfig(**raw.get("audio", {})),
            wake_word=WakeWordConfig(**raw.get("wake_word", {})),
            vad=VADConfig(**raw.get("vad", {})),
            stt=STTConfig(
                **raw.get("stt", {}),
            ),
            brains=BrainsConfig(
                routing_mode=routing_mode,
                cloud_fallback_enabled=cloud_fallback_enabled,
                brain1_primary=_brain_model(brains_raw["brain1_primary"]),
                brain1_fallback=_brain_model(brains_raw["brain1_fallback"]),
                brain2_specialist=GeminiConfig(**gemini_raw),
                brain3_failsafe=_brain_model(brains_raw["brain3_failsafe"]),
            ),
            ollama=OllamaConfig(
                **{
                    **raw.get("ollama", {}),
                    "compaction_brain": raw.get("ollama", {}).get(
                        "compaction_brain", "flash"
                    ),
                }
            ),
            tts=TTSConfig(
                **raw.get("tts", {}),
                deepgram_api_key=deepgram_api_key,
            ),
            streaming=StreamingConfig(**raw.get("streaming", {})),
            orchestrator=OrchestratorConfig(
                unrecoverable_error_apology=str(raw.get("orchestrator", {}).get("unrecoverable_error_apology", "I ran into a problem and had to reset. Try again.")),
                state_idle_timeout_ms=int(raw.get("orchestrator", {}).get("state_idle_timeout_ms", 0)),
                post_speak_settle_ms=int(raw.get("orchestrator", {}).get("post_speak_settle_ms", 300)),
                barge_in_min_speech_ms=int(raw.get("orchestrator", {}).get("barge_in_min_speech_ms", 320)),
                follow_up_min_confidence=float(raw.get("orchestrator", {}).get("follow_up_min_confidence", -1.0)),
                barge_in_duck_volume=float(raw.get("orchestrator", {}).get("barge_in_duck_volume", 0.1)),
                push_to_talk_enabled=bool(raw.get("orchestrator", {}).get("push_to_talk_enabled", True)),
                enrollment_timeout_s=float(raw.get("orchestrator", {}).get("enrollment_timeout_s", 30.0)),
                observer_shutdown_grace_s=float(raw.get("orchestrator", {}).get("observer_shutdown_grace_s", 5.0)),
                follow_up=FollowUpConfig(**raw.get("orchestrator", {}).get("follow_up", {})),
            ),
            prompts=PromptsConfig(**raw.get("prompts", {})),
            paths=PathsConfig(**raw.get("paths", {})),
            verification=VerificationConfig(**raw.get("verification", {})),
            aec=AECConfig(**raw.get("aec", {})),
            conversation=ConversationConfig(**raw.get("conversation", {})),
            observability=ObservabilityConfig(**raw.get("observability", {})),
            search=SearchConfig(**search_raw),
            tools=ToolsConfig(**raw.get("tools", {})),
            raw=raw,
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"missing/invalid config key: {e}", spoken="Configuration file is incomplete.")

    return cfg
